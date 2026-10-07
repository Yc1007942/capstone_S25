# CPU anomaly detection experiments

Compare generative VLMs, text/image similarity, and frozen vision features for
smoking, loitering, unattended items, littering, and rough sleeping. The scripts
run inference on CPU, one model at a time. No model weights are trained.

Use Python 3.10–3.12 (3.12 recommended). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Alternatively, create the environment with `uv venv --python 3.12`, then use
`uv pip install` for the two install commands. The CPU wheel index avoids
installing CUDA dependencies. Video preparation also requires `ffmpeg` on PATH.

The source is this [Google Drive folder](https://drive.google.com/drive/folders/1sj9yoyBsmaqMxpnueemZOoPDNrb4p54j).
It contains nested datasets, videos, ZIP archives, and documents. Preview before
downloading, or fetch just one category to start:

```bash
python scripts/fetch_data.py --list-only
python scripts/fetch_data.py --category smoking --list-only
python scripts/fetch_data.py --category smoking --extract-zip --speed-mbps 20
# Download all categories instead:
python scripts/fetch_data.py --extract-zip
```

Downloads resume and completed files are skipped locally before any file-download
request, including valid empty annotation files. Incomplete `.part` files are
resumed by gdown. `--retries 2` retries
transient transfer failures per file. Public-link retrieval errors stop that file
without repeated requests. `--output` changes the destination (default `data/raw`);
a category download adds its category directory. `--url` accepts a different
public Drive folder. Drive must permit public downloads; quota or permission
errors return a nonzero exit code. ZIP extraction is explicit and rejects paths
outside the extraction directory. Other archives must be extracted separately.
Documents are downloaded as supporting material and are not benchmark inputs.

If a file cannot be retrieved, the script prints its exact filename and browser
link and writes successful files and failures to `download_receipt.json` in the
output directory. The folder-discovery messages do not mean files were downloaded.
Try the failed file's browser link: if downloading it fails there too, check the
file's sharing/download permissions or wait for Drive quota recovery. A public
folder listing does not guarantee every file is downloadable.

To fetch the remaining accessible files while recording failures:

```bash
python scripts/fetch_data.py --extract-zip --continue-on-error
```

This still returns exit code 1 when any file fails. It continues past individual
failures but stops after five consecutive failed download requests. Continuing
through thousands of blocked requests does not fix Drive access. Use
`--max-consecutive-errors 0` only if you deliberately want to disable that stop.
`--request-interval 1` spaces file-download attempts by at least one second (the
default); it does not guarantee that Google will allow the requests. gdown may
make several HTTP requests inside an attempt. `--speed-mbps` limits bandwidth,
which is separate from request pacing.

Rerunning resumes transfers and skips completed files. Each successfully
downloaded ZIP is extracted even when another file fails. For a smaller retry,
select a category with `--category`. To reduce thousands of file transfers to an
archive download for an initial smoking experiment, keep the same output directory:

```bash
python scripts/fetch_data.py --include "*smoking_cctv_image.v1i.coco.zip" --extract-zip
```

`--include` accepts case-insensitive glob patterns against relative paths, with
Windows and Unix separators normalized. Repeat it to include more patterns.
For image-only inference, `--include "*.jpg" --include "*.png"` excludes the
individual annotation text files. Files under YOLO `labels/` contain one object
per row as `class x_center y_center width height`, with coordinates normalized
to the image dimensions. Their class IDs must be interpreted using the dataset's
class-name mapping, typically in `data.yaml`; a class ID alone does not identify
one of this project's anomalies. Retain these annotations for detection evaluation.
The current image-classification benchmark uses reviewed CSV labels and does not
automatically import YOLO box annotations.
For datasets available only as loose files, you can instead download their folder
as a ZIP in the Drive browser or ask the owner to provide an archive. gdown's
individual-file route still needs a public or authenticated download for each file.

If a file downloads in your signed-in browser while anonymous gdown fails, use
your own account's cookies. Upgrade the download dependency first:

```bash
python -m pip install --upgrade gdown==6.4.1
```

When Firefox is installed and signed into the account that can download the file,
this diagnostic resolves one failing file and imports Google's cookies into
gdown's cache without downloading the dataset:

```bash
python -m gdown --cookies-from-browser firefox --json "https://drive.google.com/uc?id=1yk_c6oEfVlhLGqinmutIK4RjpvF55egR"
python scripts/fetch_data.py --extract-zip --continue-on-error --use-cookies
```

On Windows, current Chrome cookies may not be decryptable by gdown; the
[gdown FAQ](https://github.com/wkentaro/gdown#faq) recommends Firefox. Alternatively,
export your own Google cookies in Netscape format and provide their existing path:

```bash
python scripts/fetch_data.py --extract-zip --continue-on-error --cookies-file /path/to/cookies.txt
```

The script imports no browser cookies automatically. Cookie use is explicit;
`--cookies-file` enables it for the supplied file, while `--use-cookies` loads
the default `~/.cache/gdown/cookies.txt`. These files contain session credentials:
keep them private and outside this repository. Authentication and pacing do not
override a file owner's download restrictions or an exhausted download quota.

Create an image manifest, optionally sampling videos at one frame every five
seconds, at most 32 frames per video:

```bash
python scripts/init.py --sample-videos
# For a directory containing only images:
python scripts/init.py --data-dir data/raw --manifest data/manifest.csv
```

Frames are saved in `data/frames`. `--fps`, `--max-frames-per-video`, `--max-side`,
and `--threads` control extraction cost. The script refuses to overwrite an
existing manifest or frame sequence, preserving reviewed labels. Use fresh paths
when changing extraction settings.

Review `data/manifest.csv` before evaluating accuracy. Paths are relative to the
manifest's directory. `suggested_label` is a folder hint only; `label` starts
empty because category folders can contain normal frames and unrelated examples.
COCO/Roboflow bounding boxes and temporal annotations are not imported automatically.
Assign a verified image-level label or leave it empty for timing-only experiments:

```csv
path,label,split,group_id,suggested_label
raw/smoking/example.jpg,smoking,test,scene_001,smoking
raw/normal/walk.jpg,normal,test,scene_002,normal
raw/reference/smoke.jpg,smoking,reference,scene_003,smoking
```

Allowed labels and prompts are in `scripts/anomalies.json`: `normal`, `smoking`,
`loitering`, `unattended_items`, `littering`, and `rough_sleeping`. Edit descriptions
to match your operational definitions, or supply another JSON with `--definitions`.
This first experiment assigns one best category per image; it does not provide
bounding boxes or multiple simultaneous labels.

Set `split` to `test`, `reference`, or `ignore`. Vision baselines require at least
one verified `reference` image for **every configured class**, including normal.
Use several varied examples if possible. Keep all frames from the same video or
scene in the same split; `group_id` is the source video for sampled frames, and the
benchmark rejects groups shared between reference and test. For related still
images, manually assign the same `group_id`. Add normal test examples to measure
false positives; ROC-AUC is unavailable with only one binary class.

List models and run a small comparison:

```bash
python scripts/benchmark.py --list-models
python scripts/benchmark.py --manifest data/manifest.csv \
  --models clip-vit-b32 smolvlm-256m --limit 20 --threads 2 --duty-cycle 0.5

# Once reference images are labeled:
python scripts/benchmark.py --manifest data/manifest.csv \
  --models mobilenet-v3-small resnet18 dinov2-small --limit 20

# Compare every registered model:
python scripts/benchmark.py --manifest data/manifest.csv --models all \
  --limit 50 --repeats 3 --timeout-seconds 1800 --max-rss-mb 4096
```

| Model name | Approach | Reference images |
| --- | --- | --- |
| `clip-vit-b32`, `clip-vit-b16` | CLIP similarity to anomaly descriptions | No |
| `smolvlm-256m`, `smolvlm-500m` | SmolVLM, deterministic JSON label generation | No |
| `mobilenet-v3-small`, `resnet18` | Frozen timm features + nearest labeled reference | Every class |
| `dinov2-small` | Frozen DINOv2 features + nearest labeled reference | Every class |

Model adapters follow the [CLIP API](https://huggingface.co/docs/transformers/v4.57.1/model_doc/clip),
[SmolVLM model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
[DINOv2 API](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov2), and
[timm feature extraction API](https://huggingface.co/docs/timm/feature_extraction).
Add checkpoints/adapters in `scripts/models.py` to extend the comparison.

`--threads` limits PyTorch and numerical-library threads. `--duty-cycle 0.5`
sleeps for the duration of each inference or reference operation; `1` disables
that pacing. `--max-samples-per-second 1` adds a start-to-start operation rate cap.
These are cooperative limits: a forward pass can occupy all configured threads
until it finishes. Model loading is not duty-cycle paced. Optional `--cpu-cores`
sets Linux affinity to available CPU IDs. `--timeout-seconds` bounds each worker's
entire run, including downloads, loading, references, warmups, and sleeps.
`--max-rss-mb` terminates workers that exceed the sampled RSS limit; it is not a
hard OS memory reservation and brief spikes may be missed.

All models use CPU float32 and eager attention. Input images are capped at 512
pixels on the longest edge by default; VLM output is capped at 32 new tokens.
Small objects such as cigarettes may need higher resolution. First runs download
weights; use `--cache-dir models` for a project-local cache and `--offline` after
warming it. `--limit 0` selects all test images. Selection is shuffled with
`--seed 42`, shared by every model, and recorded for reproducibility.

Each run creates `results/<UTC timestamp>/`:

- `comparison.csv` and `results.json`: latency, throughput, RSS, failures, and metrics.
- `run.json`: selected samples, definitions, settings, input hashes, and host details.
- `<model>/predictions.jsonl`: streamed predictions, scores, raw VLM responses, and errors.
- `<model>/summary.json` and `worker.log`: per-model measurements and diagnostics.

Latency includes image decoding, preprocessing, model inference, and output
parsing. It excludes model loading, reference preparation, warmups, and deliberate
sleep. Active throughput uses completed inference time; wall throughput includes
pacing and failed attempts. RSS is sampled across worker setup, loading, and
inference. Each model gets a fresh process so memory from earlier models is freed.

Accuracy, per-class precision/recall/F1, binary anomaly F1, confusion matrices, and
ROC-AUC use verified test labels from the first repeat. `accuracy`, `macro_f1`,
and binary metrics describe valid predictions; inspect `prediction_coverage` and
`accuracy_including_failures` alongside them. VLM `unknown` or malformed replies
are abstentions, never automatic normal predictions. CLIP scores are relative
candidate weights, not calibrated probabilities; vision scores are cosine
similarities and anomaly scores are the anomaly-vs-normal similarity margin.
VLMs have no numeric anomaly score, so their ROC-AUC is unavailable. A failed model
does not stop the remaining comparison; partial/failed runs return exit code 1.

Loitering, littering actions, and whether an item is unattended depend on time.
This frame-level benchmark tests visible cues and cannot establish persistence,
ownership, or an action from a single still. Video-level labels should not be
copied onto every sampled frame. Use reviewed frame labels for this experiment;
temporal windows or tracking will be needed to evaluate those event definitions.
For rough sleeping, assess visible posture/bedding without inferring housing status.

Run local checks without downloading weights:

```bash
python -m unittest discover -s tests -v
```
