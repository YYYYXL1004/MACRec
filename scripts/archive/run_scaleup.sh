#!/bin/bash
# T5 Scale-up 实验: Instruments 数据集
# d_model=256, d_ff=2048, num_heads=8, num_layers=4
# 基于 E1 (ST+concat+CAQ) 的 RQVAE codes
# 先不加CPA，纯 scale-up baseline

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_BASE="R1_E1_st-concat-caq"

INDEX_FILE=".index_lemb_${EXP_BASE}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_BASE}.json"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
VALID_TASK=seqrec

cd /data/yaoxianglin/MACRec

EXP_NAME="scaleup-d256-L4"
OUTPUT_DIR="./log/${DATASET}-${EXP_NAME}"
# 使用大模型配置
BASE_MODEL="./config/ckpt_large"
TRAIN_GPUS="4,5"
PORT=29630

BATCH_SIZE=1024

echo "============================================"
echo " T5 Scale-up: ${EXP_NAME}"
echo " d_model=256, d_ff=2048, heads=8, layers=4"
echo " batch_size=${BATCH_SIZE}"
echo " 开始时间: $(date)"
echo "============================================"

mkdir -p $OUTPUT_DIR

# 训练
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --per_device_batch_size $BATCH_SIZE \
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
    --valid_task $VALID_TASK 2>&1 | tee $OUTPUT_DIR/train.log

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

echo "[全流程完成] $(date)"
echo "============ 结果 ============"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
