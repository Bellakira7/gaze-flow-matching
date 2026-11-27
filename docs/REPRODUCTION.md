# Research reproduction

## Main-model training

The published main row is `flow_eye_d6_s0/epoch_14.pt`: 15 epochs with zero-based epoch numbering, seed 0, frozen DINOv2 ViT-L/14, global batch size 64, Adam, learning rate 3e-4, cosine scheduling, and gradient clipping at 5. The final epoch is used. See `configs/train_args.json`.

The training dataset is the ShijianDeng GazeFollow mirror. Its images have eye points but no head boxes; the converter constructs eye-centered square boxes with side length 0.12 sqrt(W H). A deterministic image-content split reserves a 3% training holdout. This is different from using annotated test head boxes.

Obtain the dataset separately, place the training parquet at `data/hf_shijiandeng_train/train.parquet`, and run:

```bash
python tools/build_gazefollow_train_from_sd.py
export DINOV2_WEIGHTS=/path/to/dinov2_vitl14_pretrain.pth
torchrun --nproc_per_node 4 gazelle_online/train_heads_eye.py \
  --head flow --head_kw '{"depth": 6}' \
  --backbone dinov2_vitl14 --data_path data/gazefollow_train_sd \
  --exp_name flow_eye_d6_s0 --exp_root experiments \
  --max_epochs 15 --batch_size 64 --lr 0.0003 --clip 5 --seed 0
```

Training evaluates the training holdout after each epoch and writes checkpoint/metric artifacts. The release does not launch or rerun training automatically.

## GazeFollow evaluation

Convert the vikhyatk GazeFollow test parquet:

```bash
python tools/build_gazefollow_test_from_hf.py /path/to/test.parquet data/gazefollow_test_hf
```

`gazelle_cached/extract_features.py --help` describes scene feature extraction. The original test path expects `feats/gf_test_vitl14/test_orig.npy` in the order of `test_preprocessed.json`. Local crops use the same image transform and frozen backbone online.

With those inputs prepared, the main-model prediction command is:

```bash
mkdir -p results/g2
python gazelle_online/predict_test_ray.py \
  --variant eye --head_kw '{"depth": 6}' \
  --ckpt checkpoints/flowET_d6_s0_epoch14.pt --box gt \
  --n 10000 --sigma_h 1.0 \
  --sample_kw '{"solver":"heun","steps":16,"gamma":3.0}' \
  --out results/g2/pred_flowET_d6_gt.npz
```

The complete G2 bundle/gate additionally needs the G1 separated-mode mask at `results/g1/per_instance.npz`, the per-instance annotations, and matching baseline prediction bundles. Those large experiment artifacts are not part of this release. `setmetrics/g2_eval.py` and `gazelle_online/g2_read.py` contain the preserved evaluation rules; `results/paper/` contains archived results.

## VideoAttentionTarget and supplemental experiments

Both downloadable demos apply the **GazeFollow main checkpoint** frame by frame. The original v1.0.0 demo uses supplied head boxes; v1.1.0 uses YuNet face-derived head boxes and automatic geometry-based tracking. Only head-box geometry is temporally filtered; gaze predictions are not. Neither automatic detection nor track identity accuracy has been benchmarked here. It is not the VAT-finetuned checkpoint used by the separate current-frame VAT experiment. Forecasting, current-frame VAT, ray variants, and other ablations remain in `gazelle_online/` as separate research entry points.

VAT comparative scoring uses shared tools that also served the Gaze-R1 experiments. These modules are included to preserve imports, but their VLM training/inference commands are not required for the paper's flow model or this demo.

## Portability and validation

Path constants now resolve from each script's repository location. `DINOV2_WEIGHTS` supplies a local frozen backbone; otherwise the bundled DINOv2 loader uses its official pretrained-weight URL. The original source hashes are preserved in `provenance/source_manifest.json`.

The original protocol documents retain their bytes and checksums. Packaging does not establish a fresh full-benchmark reproduction. The release validation checks parsing, dependency imports, checkpoint loading, and video inference; its concrete results are recorded in `provenance/validation.json`.

Automatic video validation, including the exact preset and limitations, is recorded in [AUTO-HEAD-20261127](../experiments/AUTO-HEAD-20261127.md). CPU tracking checks can be run with `python -m unittest discover -s tests -v`.
