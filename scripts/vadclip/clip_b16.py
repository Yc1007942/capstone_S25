"""Clean reimplementation of OpenAI CLIP ViT-B/16 (the exact backbone VadCLIP uses).

This is a from-scratch port of the official ``openai/clip`` ViT-B/16 model, kept to
only what VadCLIP needs:

  * ``encode_image(image) -> [N, 512]``   raw (unnormalised) patch features,
  * ``encode_token(token) -> [N, 77, 512]``  the authors' raw token-embedding lookup,
  * ``encode_text(embeddings, tokens) -> [N, 512]``  the authors' *modified* text
    encoder that consumes pre-built embeddings (not token ids) and reads out at the
    eot position -- this is what lets VadCLIP splice real tokens into a learnable
    prompt template.

The architecture mirrors ``clip/model.py`` from the released repo; weights are the
official OpenAI ViT-B/16 checkpoint (SHA-256 verified at download time). Token ids
come from HuggingFace's ``openai/clip-vit-base-patch16`` tokenizer, which is built
from the same BPE vocabulary and uses the identical special-token ids
(SOT=49406, EOT=49407), so they are drop-in compatible with this model.

No image content is ever displayed; ``encode_image`` runs on tensors only.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (kept for faithfulness)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Text-tower block (no padding mask -- the context is always full length)."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid**2]
        x = x.permute(0, 2, 1)  # [*, grid**2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size, transformer_width,
                 transformer_heads, transformer_layers):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_heads, output_dim=embed_dim)

        self.transformer = Transformer(
            width=transformer_width, layers=transformer_layers, heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # log(1/0.07)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Raw (unnormalised) patch features [N, embed_dim]."""
        return self.visual(image.type(self.dtype))

    def encode_token(self, token: torch.Tensor) -> torch.Tensor:
        """Authors' raw token-embedding lookup: [N, 77] ids -> [N, 77, width]."""
        return self.token_embedding(token)

    def encode_text(self, text: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """Authors' modified encoder: consumes pre-built embeddings (not ids).

        ``text`` is [N, 77, width] of (possibly spliced) embeddings; ``token`` is the
        matching [N, 77] id tensor used only to locate the eot position for readout.
        Returns unnormalised features [N, embed_dim].
        """
        x = text.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # read out at the eot position (eot is the highest id in each sequence)
        x = x[torch.arange(x.shape[0]), token.argmax(dim=-1)] @ self.text_projection
        return x


def build_model(state_dict: dict) -> CLIP:
    """Reconstruct the ViT-B/16 architecture from an official state_dict."""
    assert "visual.proj" in state_dict, "expected a ViT checkpoint (has visual.proj)"
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    sd = {k: v for k, v in state_dict.items() if k not in ("input_resolution", "context_length", "vocab_size")}
    model.load_state_dict(sd)
    return model.eval()


# --- image preprocessing (official CLIP ViT-B/16 transform) -------------------

_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def make_preprocess(n_px: int = 224):
    """Return a callable PIL.Image -> [3, n_px, n_px] tensor (official CLIP transform)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN.flatten().tolist(), std=_STD.flatten().tolist()),
    ])


# --- tokenizer (OpenAI-compatible ids via HuggingFace) ------------------------

_tokenizer = None


def get_tokenizer():
    """Lazily load the OpenAI-CLIP-compatible BPE tokenizer (cached)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import CLIPTokenizer
        _tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    return _tokenizer


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.Tensor:
    """Tokenize to [N, 77] OpenAI-compatible ids (SOT=49406 at pos 0, EOT=49407).

    Pads with the eot id (49407), matching openai/clip exactly; ``argmax`` over a row
    therefore returns the position of the *first* eot -- i.e. the real end-of-text.
    """
    if isinstance(texts, str):
        texts = [texts]
    tok = get_tokenizer()
    out = tok(list(texts), padding="max_length", max_length=context_length,
              truncation=True, return_tensors="pt")["input_ids"]
    return out


def load_clip_b16(weights_path: str, device: torch.device) -> CLIP:
    """Load the official OpenAI ViT-B/16 weights into a clean CLIP module.

    The released ``ViT-B-16.pt`` is a TorchScript archive (a ``RecursiveScriptModule``), not a
    plain state_dict -- PyTorch's loader even warns it "looks like a TorchScript archive". We load
    it as-is and pull out its ``state_dict()`` to rebuild our clean reimplementation, which adds the
    VadCLIP-modified ``encode_text(embeddings, tokens)`` that the ScriptModule does not expose. The
    file is SHA-256 verified against the official OpenAI hash before use (trusted source), so loading
    with ``weights_only=False`` is safe here.
    """
    obj = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = obj if isinstance(obj, dict) else obj.state_dict()
    model = build_model(state_dict).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
