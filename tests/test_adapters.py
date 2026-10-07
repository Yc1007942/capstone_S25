"""CPU adapter contracts and equivalence to native model scoring."""

from __future__ import annotations

import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.models import (
    DEFAULT_MODELS,
    MODELS,
    MiniCPMAdapter,
    OpenCLIPAdapter,
    SigLIPAdapter,
)

DEFINITIONS = {"normal": "Ordinary walking", "smoking": "A Person Smoking"}
ML_AVAILABLE = all(
    importlib.util.find_spec(name)
    for name in ("torch", "transformers", "timm", "open_clip")
)


class RegistryTests(unittest.TestCase):
    def test_default_shortlist_and_selected_checkpoints(self):
        self.assertEqual(
            DEFAULT_MODELS, ["mobileclip2-s2", "siglip2-b16-224", "smolvlm-500m"]
        )
        self.assertEqual(MODELS["minicpm-v4"].checkpoint, "openbmb/MiniCPM-V-4")
        self.assertEqual(
            MODELS["siglip2-b16-224"].checkpoint, "google/siglip2-base-patch16-224"
        )


@unittest.skipUnless(ML_AVAILABLE, "Model dependencies are not installed")
class AdapterTests(unittest.TestCase):
    def test_siglip_cached_scoring_matches_native_forward(self):
        import torch
        from PIL import Image
        from transformers import SiglipConfig, SiglipModel

        torch.manual_seed(123)
        config = SiglipConfig(
            text_config={
                "vocab_size": 10,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 64,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 16,
                "patch_size": 4,
            },
        )
        native = SiglipModel(config).eval()
        input_ids = torch.tensor([[1] * 63 + [2], [1] * 63 + [3]])
        pixels = torch.rand(1, 3, 16, 16)
        text_calls = []

        class Processor:
            def __call__(self, **kwargs):
                if "text" in kwargs:
                    text_calls.append(kwargs)
                    return {"input_ids": input_ids}
                return {"pixel_values": pixels}

        with (
            patch(
                "transformers.AutoProcessor.from_pretrained", return_value=Processor()
            ),
            patch("transformers.AutoModel.from_pretrained", return_value=native),
        ):
            adapter = SigLIPAdapter(
                MODELS["siglip2-b16-224"], DEFINITIONS, {"local_files_only": True}
            )
        image = Image.new("RGB", (16, 16))
        prediction = adapter.predict(image)
        adapter.predict(image)
        with torch.inference_mode():
            expected = native(
                input_ids=input_ids, pixel_values=pixels
            ).logits_per_image[0]
        torch.testing.assert_close(
            torch.tensor(list(prediction.scores.values())), expected.sigmoid()
        )
        self.assertEqual(prediction.label, list(DEFINITIONS)[expected.argmax().item()])
        self.assertAlmostEqual(
            prediction.anomaly_score, (expected[1] - expected[0]).item(), places=5
        )
        self.assertEqual(len(text_calls), 1)
        self.assertEqual(
            text_calls[0]["text"],
            [description.lower() for description in DEFINITIONS.values()],
        )
        self.assertEqual(text_calls[0]["padding"], "max_length")
        self.assertEqual(text_calls[0]["max_length"], 64)

    def test_mobileclip_fuses_eval_model_and_caches_text(self):
        import torch
        from PIL import Image

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10)))
                self.text_calls = 0

            def encode_text(self, tokens, normalize):
                self.text_calls += 1
                assert not self.training
                return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

            def encode_image(self, image, normalize):
                assert not self.training
                return torch.nn.functional.normalize(torch.tensor([[0.2, 0.8]]), dim=-1)

        native = Model()
        factory = Mock(
            return_value=(native, None, lambda image: torch.zeros(3, 16, 16))
        )

        def fuse(model, *, inplace):
            self.assertFalse(model.training)
            self.assertTrue(inplace)
            return model

        with (
            patch("open_clip.create_model_and_transforms", factory),
            patch(
                "open_clip.get_tokenizer",
                return_value=lambda descriptions: torch.ones(2, 2, dtype=torch.long),
            ),
            patch("timm.utils.reparameterize_model", side_effect=fuse) as fusion,
        ):
            adapter = OpenCLIPAdapter(
                MODELS["mobileclip2-s2"], DEFINITIONS, {"cache_dir": "/tmp/cache"}
            )
        prediction = adapter.predict(Image.new("RGB", (16, 16)))
        adapter.predict(Image.new("RGB", (16, 16)))
        self.assertEqual(prediction.label, "smoking")
        self.assertAlmostEqual(sum(prediction.scores.values()), 1, places=6)
        self.assertEqual(native.text_calls, 1)
        self.assertEqual(factory.call_args.kwargs["device"], "cpu")
        self.assertEqual(factory.call_args.kwargs["precision"], "fp32")
        fusion.assert_called_once()

    def test_minicpm_uses_cached_processor_and_bounded_greedy_chat(self):
        import torch
        from PIL import Image

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.chat = Mock(return_value='{"label":"smoking"}')

        native = Model()
        processor = SimpleNamespace(tokenizer=object())
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
            patch(
                "transformers.AutoModel.from_pretrained", return_value=native
            ) as loader,
        ):
            adapter = MiniCPMAdapter(
                MODELS["minicpm-v4"], DEFINITIONS, {"local_files_only": True}, 32
            )
        image = Image.new("RGB", (16, 16))
        prediction = adapter.predict(image)
        self.assertEqual(prediction.label, "smoking")
        self.assertEqual(loader.call_args.kwargs["dtype"], torch.float32)
        self.assertEqual(loader.call_args.kwargs["attn_implementation"], "sdpa")
        self.assertTrue(loader.call_args.kwargs["trust_remote_code"])
        options = native.chat.call_args.kwargs
        self.assertIs(options["processor"], processor)
        self.assertIs(options["msgs"][0]["content"][0], image)
        self.assertFalse(options["sampling"])
        self.assertEqual(options["num_beams"], 1)
        self.assertEqual(options["max_new_tokens"], 32)
        self.assertEqual(options["max_slice_nums"], 1)
        native.chat.return_value = "insufficient evidence"
        self.assertIsNone(adapter.predict(image).label)


if __name__ == "__main__":
    unittest.main()
