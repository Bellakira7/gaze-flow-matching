#!/usr/bin/env bash
# Adapted from experiment-protocol/templates/launch_local.sh.
# This worker is started by nohup; caller checks config and progress separately.
set -euo pipefail
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT || true
set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
BASE=/cpfs01/shared/public/groups/closeloop/ace.yang/gaze/demos/AUTO-HEAD-20261127
SOURCE=/cpfs01/shared/public/groups/closeloop/ace.yang/gaze/demos/FLOW-VIDEO-20261127/source_continuous.mp4
CKPT=/cpfs01/shared/public/users/ace.yang/gaze/experiments/flow_eye_d6_s0/epoch_14.pt
BACKBONE=/cpfs01/shared/public/users/yuezhang.lv/RoMa/dinov2_vitl14_pretrain.pth
cd "$BASE/code"
trap 'echo $? > "$BASE/exit_code"' EXIT
ARGS=(--checkpoint "$CKPT" --backbone-weights "$BACKBONE")
/usr/local/bin/python -m unittest discover -s tests -v
/usr/local/bin/python demo.py --video "$SOURCE" "${ARGS[@]}" --limit 2 --out "$BASE/auto_smoke"
/usr/local/bin/python demo.py --video "$SOURCE" "${ARGS[@]}" --head-boxes assets/head_boxes.csv --limit 2 --out "$BASE/supplied_smoke"
ffmpeg -v error -f lavfi -i color=c=black:s=640x360:r=30 -frames:v 2 -c:v libx264 -pix_fmt yuv420p "$BASE/blank.mp4"
/usr/local/bin/python demo.py --video "$BASE/blank.mp4" "${ARGS[@]}" --out "$BASE/blank_smoke"
/usr/local/bin/python demo.py --video "$SOURCE" "${ARGS[@]}" --out "$BASE/full"
