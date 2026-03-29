#!/bin/bash
# 序列增强实验1: Instruments E1, item_dropout=0.2
# 双卡 DDP GPU 0,1 (与 aug2 共享)
# 用法: screen -dmS aug1 bash scripts/active/run_aug1_inst.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_BASE="R1_E1_st-concat-caq"
EXP_NAME="aug-drop0.2"
OUTPUT_DIR="./log/${DATASET}-${EXP_NAME}"
TRAIN_GPUS="0,1"
PORT=29701

INDEX_FILE=".index_lemb_${EXP_BASE}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_BASE}.json"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
VALID_TASK=seqrec

cd /data/yaoxianglin/MACRec

echo "============================================"
echo " 序列增强实验: ${EXP_NAME}"
echo " aug_item_dropout=0.2"
echo " GPU: ${TRAIN_GPUS}, PORT: ${PORT}"
echo " 开始时间: $(date)"
echo "============================================"

mkdir -p $OUTPUT_DIR

# 检测可恢复的 checkpoint
RESUME_ARG=""
LATEST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ] && [ -s "$LATEST_CKPT/model.safetensors" ]; then
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
    echo "[恢复训练] 从 ${LATEST_CKPT} 恢复"
fi

# T5 训练（双卡 DDP）
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
    --aug_item_dropout 0.2 \
    $RESUME_ARG 2>&1 | tee -a $OUTPUT_DIR/train.log

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

echo "============================================"
echo " ${EXP_NAME} 全流程完成: $(date)"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
