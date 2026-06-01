#!/bin/bash
# UNGER baseline (text+collab unified single code) — single-branch seqrec.
# Usage: bash run_unger.sh <gpu_id> <dataset> <port>
set -e
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPU=$1
DATASET=$2
PORT=$3
SAVE_NAME=unger
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $PROJECT_DIR

Index_file=.index_lemb_${SAVE_NAME}.json
OUTPUT_DIR=./log/repro/${DATASET}-${SAVE_NAME}
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

echo "[UNGER] $DATASET on GPU $GPU port $PORT -> $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --base_model ./config/ckpt \
    --per_device_batch_size 512 \
    --learning_rate 1e-3 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Index_file \
    --tasks seqrec \
    --valid_task seqrec > $log_file 2>&1

results_file=$OUTPUT_DIR/results_seqrec_20.json
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT test_ddp.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Index_file \
    --test_task seqrec \
    --results_file $results_file \
    --filter_items >> $log_file 2>&1

echo "[UNGER] $DATASET DONE. Results:"
python3 -c "import json;print(json.load(open('$results_file'))['mean_results'])"
