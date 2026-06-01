#!/bin/bash
# MSCGRec baseline — 3 independent modality codes, multimodal history, collab-target decoding.
# Usage: bash run_mscgrec.sh <gpu_id> <dataset> <port>
set -e
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPU=$1
DATASET=$2
PORT=$3
PROJECT_DIR=/sda/data/yaoxianglin/MACRec
cd $PROJECT_DIR

DDIR=data/$DATASET
TEXT_IDX=.index_lemb_mscgrec.json
IMG_IDX=.index_vitemb_mscgrec.json
COLLAB_IDX=.index_collab_mscgrec.json

# --- Step 1: generate 3 independent modality codes (residual RQ) ---
echo "[MSCGRec] $DATASET: generating modality codes"
cd analysis/repro
python residual_kmeans.py --emb_path ../../$DDIR/$DATASET.emb-llama-td.npy   --out_path ../../$DDIR/$DATASET$TEXT_IDX   --prefixes abcd
python residual_kmeans.py --emb_path ../../$DDIR/$DATASET.emb-ViT-L-14.npy   --out_path ../../$DDIR/$DATASET$IMG_IDX    --prefixes ABCD
python residual_kmeans.py --emb_path ../../$DDIR/$DATASET.emb-collab-256.npy --out_path ../../$DDIR/$DATASET$COLLAB_IDX --prefixes PQRS
cd $PROJECT_DIR

OUTPUT_DIR=./log/repro/${DATASET}-mscgrec
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

# --- Step 2: train ---
echo "[MSCGRec] $DATASET on GPU $GPU port $PORT -> $OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT finetune_mscgrec.py \
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
    --patient 10 \
    --index_file $TEXT_IDX \
    --image_index_file $IMG_IDX \
    --collab_index_file $COLLAB_IDX \
    --valid_task seqrec > $log_file 2>&1

# --- Step 3: evaluate ---
results_file=$OUTPUT_DIR/results_mscgrec_20.json
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT test_mscgrec.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $TEXT_IDX \
    --image_index_file $IMG_IDX \
    --collab_index_file $COLLAB_IDX \
    --test_task seqrec \
    --results_file $results_file \
    --filter_items >> $log_file 2>&1

echo "[MSCGRec] $DATASET DONE. Results:"
python3 -c "import json;print(json.load(open('$results_file'))['mean_results'])"
