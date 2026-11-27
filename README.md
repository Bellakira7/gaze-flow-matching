# Learning Gaze Distributions with Conditional Flow Matching

Private research release for the paper **Learning Gaze Distributions with Conditional Flow Matching**.

The released model, **flowET_d6**, combines a frozen DINOv2 ViT-L/14 backbone, scene and local head-region tokens, and a six-layer conditional flow decoder. It predicts a spatial gaze distribution and five weighted target hypotheses.

## Checkpoint and continuous-video demo

| Artifact | Download |
| --- | --- |
| Paper checkpoint: seed 0, epoch 14 | [flowET_d6_s0_epoch14.pt](https://github.com/Bellakira7/gaze-flow-matching/releases/download/v1.0.0/flowET_d6_s0_epoch14.pt) — 38,779,814 bytes |
| Continuous-video predictions | [Demo MP4](assets/flowET_d6_continuous_demo.mp4) · [Release download](https://github.com/Bellakira7/gaze-flow-matching/releases/download/v1.0.0/flowET_d6_continuous_demo.mp4) |
| Input video | [source_continuous.mp4](https://github.com/Bellakira7/gaze-flow-matching/releases/download/v1.0.0/source_continuous.mp4) |
| Input head track | [head_boxes.csv](assets/head_boxes.csv) |

[![Continuous-video demo](assets/demo_preview.jpg)](assets/flowET_d6_continuous_demo.mp4)

The demo contains **360 consecutive frames**, played at **30 fps for 12 seconds**, from the VideoAttentionTarget MLB Interview clip. Every frame was independently processed with the released checkpoint. The cyan box identifies the right-hand interviewer; the heatmap and gold targets are model predictions. Head boxes come from the dataset's person track. Ground-truth gaze is not supplied, and there is no temporal smoothing. Playback frame rate is not inference speed. The source dataset provides extracted frames, so the demo has no audio.

This is the fixed seed-0 main model used by the paper. The `.pt` file contains the trainable trunk and flow decoder. The frozen **DINOv2 ViT-L/14 weights are separate** and loaded from an official download/cache or an explicitly supplied file.

Checkpoint SHA-256:

```text
200072de520ca356e69591121a63e3ef4979b046b08d4166fe3564a5e4503657
```

Downloads require access to this private repository. Sign in to GitHub and download the release assets, or use an authenticated GitHub CLI:

```bash
gh release download v1.0.0 --repo Bellakira7/gaze-flow-matching \
  --pattern flowET_d6_s0_epoch14.pt --dir checkpoints
gh release download v1.0.0 --repo Bellakira7/gaze-flow-matching \
  --pattern source_continuous.mp4 --dir assets
```

## Installation

Use Python 3.10 or newer, a CUDA-compatible GPU, and `ffmpeg`/`ffprobe` on PATH. Install the PyTorch/torchvision build suitable for your accelerator, then install the remaining dependencies:

```bash
git clone https://github.com/Bellakira7/gaze-flow-matching.git
cd gaze-flow-matching
python -m pip install -r requirements.txt
```

Gaze-LLE and DINOv2 Python sources are vendored at the revisions recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No extra source-code clone is required. `xformers` is optional; DINOv2 supports its standard PyTorch attention path.

The original continuous demo used torch 2.9.0, torchvision 0.24.0, numpy 1.26.4, scipy 1.17.1, timm 1.0.27, and a PPU-ZW810E accelerator with its CUDA-compatible vendor runtime. The release's portability check and its exact scope are recorded in [provenance/validation.json](provenance/validation.json).

## Run the continuous-video demo

After downloading the checkpoint and input video:

```bash
python demo.py \
  --video assets/source_continuous.mp4 \
  --head-boxes assets/head_boxes.csv \
  --checkpoint checkpoints/flowET_d6_s0_epoch14.pt \
  --out output/demo
```

On an offline machine, add `--backbone-weights /path/to/dinov2_vitl14_pretrain.pth`. For the research scripts, set the equivalent environment variable `DINOV2_WEIGHTS`.

The output directory contains `demo.mp4`, `predictions.npz`, preview frames, and `stats.json` with source/checkpoint hashes. The default sampler uses 10,000 points, Heun integration with 16 steps and gamma 3, followed by five-hypothesis extraction with sigma_h 1. A common random seed is reset for every frame to reduce Monte Carlo flicker; frames are still processed independently.

For your own constant-frame-rate video, supply one normalized head box per frame in this format:

```csv
frame,xmin,ymin,xmax,ymax
0,0.6734375,0.1583333,0.8398438,0.4861111
1,0.6734375,0.1583333,0.8398438,0.4861111
```

Frame indices start at zero and must be consecutive. This entry point tracks one supplied person. It does not include a head detector or an out-of-frame classifier. `--limit 2` runs a short check. The downloadable source MP4 is a lossy encoding of the original dataset JPEG frames; re-running it may differ slightly from the archived demo, which used those original JPEGs directly.

## Research code

| Directory | Contents |
| --- | --- |
| `gazelle_online/` | Flow/MDN/MCL models, local-token and ray variants, training, inference, sampler selection, ablations, forecast experiments, and paper table generation |
| `gazelle_cached/` | Backbone loading, cached-feature extraction, and heatmap baseline utilities |
| `setmetrics/` | Set-aware metrics, prediction bundles, and G2 evaluation/gating |
| `tools/` | Dataset conversion and shared scoring utilities used by the VAT comparisons; some retain legacy Gaze-R1 entry points |
| `configs/` | Main-model training arguments and frozen sampler selection |
| `prereg/` | Original G1/S1/G2 protocol documents and later clarifications |
| `results/paper/` | Archived paper tables and latency results |
| `provenance/` | Original code hashes, release validation, and demonstration metadata |

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for training and evaluation inputs. Dataset images, training checkpoints for other arms, and the large prediction archives are not bundled. Full benchmark reconstruction requires those datasets and intermediate artifacts; the release assets are sufficient for the continuous-video demo.

The original flow formulation and the later eye-token/depth variants are retained separately. The paper's selected six-layer method was developed after earlier test feedback; repeated seeds characterize that selected architecture. The archived protocol documents and tables retain this distinction.

## Source and licenses

Project paths were made relative to the repository; offline backbone weights are configurable. Original research source hashes and upstream revisions are in [provenance/source_manifest.json](provenance/source_manifest.json). Numerical model and metric implementations are retained from the research code.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the bundled upstream licenses. This is a private research release; no additional license is granted for the project-specific code, checkpoint, or underlying video.
