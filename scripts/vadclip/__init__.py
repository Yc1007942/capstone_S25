"""Faithful, CPU-clean reimplementation of VadCLIP (AAAI 2024).

VadCLIP = "Adapting Vision-Language Models for Weakly Supervised Video Anomaly
Detection" (Wu et al., AAAI 2024; github.com/nwpu-zxr/VadCLIP). The published
weights and pre-extracted CLIP features are gated behind Baidu Pan / OneDrive
access codes, so this package reimplements the *method* from the released source:

  * frozen OpenAI CLIP ViT-B/16 backbone (official weights),
  * LGT-Adapter temporal module (windowed transformer + dual graph branches),
  * dual detection branch (visual classifier + language-visual alignment),
  * the two prompt mechanisms (learnable embedding template + prefix/postfix offset).

The only deliberate deviation is supervision granularity: our anomaly corpora are
anomaly-specific (every clip contains at least one positive frame, so video-level
weak labels carry no negative signal). We therefore supervise at **frame** level
with dense per-frame ground truth -- strictly stronger than the paper's video-MIL.
See data.py and train.py for details.

No image content is ever displayed; all evaluation is programmatic inference on
extracted features, consistent with the project's no-screenshot constraint.
"""
