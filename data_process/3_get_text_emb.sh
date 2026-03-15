# 用法: bash 3_get_text_emb.sh Arts [GPU_ID]
# 支持: Arts / Games / Instruments
DATASET=${1:-Arts}
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID
python amazon_text_emb.py --dataset $DATASET