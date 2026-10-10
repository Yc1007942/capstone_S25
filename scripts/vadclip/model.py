"""Faithful CPU-clean port of VadCLIP's ``CLIPVAD`` model (AAAI 2024).

Structure mirrors ``src/model.py`` from the released repo exactly:

  * a windowed temporal transformer over frame features (LGT-Adapter),
  * two graph branches -- a content-similarity adjacency (``adj4``) and a fixed
    temporal-distance adjacency (``DistanceAdj``) -- each run through two GCN layers,
  * the dual detection head: a visual classifier (``logits1``, per-frame anomaly logit)
    and a language-visual alignment score (``logits2``, cosine similarity of each frame
    to class text features, scaled by 1/0.07),
  * the two prompt mechanisms: a learnable ``text_prompt_embeddings`` template into which
    real token embeddings are spliced at a prefix/postfix offset, fed through the frozen
    CLIP text tower via the authors' modified ``encode_text(embeddings, tokens)``.

Deliberate deviations (documented in the package docstring):
  * runs on CPU -- ``DistanceAdj`` takes an explicit device instead of hard-coding CUDA;
  * supervision is frame-level binary (see train.py), not the paper's video-MIL, because
    our anomaly corpora contain no negative clips. The architecture itself is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn

from .clip_b16 import CLIP, tokenize
from .layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Temporal-tower block (carries a padding mask through the residual stream)."""

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

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self, num_class: int, embed_dim: int = 512, visual_length: int = 256,
                 visual_width: int = 512, visual_head: int = 1, visual_layers: int = 2,
                 attn_window: int = 8, prompt_prefix: int = 10, prompt_postfix: int = 10,
                 clip_model: CLIP | None = None, device: torch.device | str = "cpu",
                 use_graph: bool = True):
        super().__init__()

        self.num_class = num_class
        self.use_graph = use_graph   # ablation knob: False -> temporal-only (no GCN branches)
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = torch.device(device)

        # --- LGT-Adapter temporal module -------------------------------------
        self.temporal = Transformer(
            width=visual_width, layers=visual_layers, heads=visual_head,
            attn_mask=self.build_attention_mask(attn_window))

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        # --- dual detection head ---------------------------------------------
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))]))
        self.classifier = nn.Linear(visual_width, 1)

        # --- frozen CLIP backbone (injected to avoid a second download) -------
        assert clip_model is not None, "pass a loaded CLIP module via clip_model="
        self.clipmodel = clip_model
        for p in self.clipmodel.parameters():
            p.requires_grad_(False)

        # --- learnable prompt / position embeddings --------------------------
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, embed_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window: int) -> torch.Tensor:
        """Block-diagonal windowed mask: each frame attends only within its window."""
        L = self.visual_length
        mask = torch.empty(L, L)
        mask.fill_(float("-inf"))
        for i in range(int(L / attn_window)):
            if (i + 1) * attn_window < L:
                mask[i * attn_window:(i + 1) * attn_window, i * attn_window:(i + 1) * attn_window] = 0
            else:
                mask[i * attn_window:L, i * attn_window:L] = 0
        return mask

    def adj4(self, x: torch.Tensor, seq_len):
        """Content-similarity adjacency: cosine sim thresholded at 0.7 + row softmax."""
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))                       # [B, T, T]
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)         # [B, T, 1]
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                adj2 = F.threshold(x2[i], 0.7, 0)
                output[i] = soft(adj2)
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = F.threshold(tmp, 0.7, 0)
                output[i, :seq_len[i], :seq_len[i]] = soft(adj2)
        return output

    def encode_video(self, images: torch.Tensor, padding_mask, lengths):
        """images: [B, T, W] raw CLIP frame features -> [B, T, W] temporal features."""
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device).unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids).permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings   # [T, B, W]

        x, _ = self.temporal((images, None))     # padded frames DO attend (faithful)
        x = x.permute(1, 0, 2)                    # -> [B, T, W]

        if not self.use_graph:
            # temporal-only ablation: skip both GCN branches, project the temporal features directly
            return self.linear(x)                 # [B, T, W] -> [B, T, W]

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1], x.device)
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)                # [B, T, 2W]
        x = self.linear(x)                        # -> [B, T, W]
        return x

    def encode_textprompt(self, text):
        """Splice real token embeddings into the learnable template at a prefix/postfix offset."""
        word_tokens = tokenize(text).to(self.device)                 # [C, 77]
        word_embedding = self.clipmodel.encode_token(word_tokens)     # [C, 77, W]
        text_embeddings = (self.text_prompt_embeddings(torch.arange(77, device=self.device))
                           .unsqueeze(0).repeat(len(text), 1, 1))     # [C, 77, W] learnable template
        text_tokens = torch.zeros(len(text), 77, dtype=torch.long, device=self.device)

        for i in range(len(text)):
            ind = int(torch.argmax(word_tokens[i], -1))              # eot position (first max id)
            assert self.prompt_prefix + ind + self.prompt_postfix < 77, \
                f"prompt too long: eot at {ind} exceeds template with prefix/postfix offsets"
            text_embeddings[i, 0] = word_embedding[i, 0]             # real SOT
            text_embeddings[i, self.prompt_prefix + 1:self.prompt_prefix + ind] = word_embedding[i, 1:ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]  # real EOT
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        return self.clipmodel.encode_text(text_embeddings, text_tokens)   # [C, embed_dim]

    def forward(self, visual: torch.Tensor, padding_mask, text, lengths):
        """visual: [B, T, W]; text: list of num_class prompt strings; lengths: [B] real frame counts.

        Returns (text_features_ori [C, W], logits1 [B, T, 1], logits2 [B, T, C] scaled by /0.07,
                 logits2_raw [B, T, C] unscaled cosine in [-1,1]). Training and inference scoring use
                 the raw cosine (softmax over classes) because the /0.07 scale saturates softmax.
        """
        visual_features = self.encode_video(visual, padding_mask, lengths)   # [B, T, W]
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)                      # [C, W]

        # attention-weighted representative visual feature (weighted by the visual branch score)
        logits_attn = logits1.permute(0, 2, 1)                                # [B, 1, T]
        visual_attn = logits_attn @ visual_features                           # [B, 1, W]
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])

        text_features = text_features_ori.unsqueeze(0).expand(visual_attn.shape[0], -1, -1)
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)              # [B, C, W]

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = (text_features / text_features.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
        logits2_raw = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype)   # [B,T,C] raw cosine in [-1,1]
        logits2 = logits2_raw / 0.07   # faithful scaled alignment score (kept for reporting only)

        return text_features_ori, logits1, logits2, logits2_raw
