# 用法: bash 4_get_image_emb.sh Arts [GPU_ID]
# 支持: Arts / Games / Instruments
DATASET=${1:-Arts}
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID
python clip_feature.py --dataset $DATASET
