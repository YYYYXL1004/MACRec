#!/bin/bash
# 轻量 CPA 实验2: cpa_weight=0.01, cosine loss
# 独立并行运行，使用 GPU 4,5
# 用法: screen -dmS lcpa2 bash run_lcpa_w001.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_BASE="R1_E1_st-concat-caq"
COLLAB_EMB="./data/${DATASET}/${DATASET}.emb-collab-256.npy"

INDEX_FILE=".index_lemb_${EXP_BASE}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_BASE}.json"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
VALID_TASK=seqrec

CPA_W=0.01
CPA_LOSS=cosine
EXP_NAME="lcpa-${CPA_LOSS}-w${CPA_W}"
OUTPUT_DIR="./log/${DATASET}-${EXP_NAME}"
TRAIN_GPUS="4,5"
PORT=29625

cd /data/yaoxianglin/MACRec

echo "============================================"
echo " ${EXP_NAME} (并行)"
echo " CPA weight=${CPA_W}, loss=${CPA_LOSS}"
echo " GPU: ${TRAIN_GPUS}"
echo " 开始时间: $(date)"
echo "============================================"

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --base_model ./config/ckpt \
    --per_device_batch_size 1024 \
    --learning_rate 1e-3 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --tasks $TASKS \
    --valid_task $VALID_TASK \
    --cpa_weight $CPA_W \
    --collab_emb_path $COLLAB_EMB \
    --cpa_loss_type $CPA_LOSS 2>&1 | tee $OUTPUT_DIR/train.log

echo "[训练完成] $(date)"

# 推理 seqrec
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_20.json \
    --save_file $OUTPUT_DIR/save_seqrec_20.json \
    --filter_items

# 推理 seqimage
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_20.json \
    --save_file $OUTPUT_DIR/save_seqimage_20.json \
    --filter_items

# Ensemble
python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams 20

echo "============ 结果 ============"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
echo "全流程完成: $(date)"
