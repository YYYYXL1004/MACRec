#!/bin/bash
# Instruments E1 RQVAE + 轻量CPA + 序列增强 + beam-50
# 全组件叠加: LCPA w=0.005 cosine + aug dropout=0.1 crop=0.3 + beam-50
# RQVAE索引: E1 (ST+concat+CAQ, 碰撞率 1.74%/1.78%)
# GPU: 1,2 双卡

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
SAVE_NAME="R1_E1_st-concat-caq"
TRAIN_GPUS="1,2"
PORT=29653

INDEX_FILE=".index_lemb_${SAVE_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${SAVE_NAME}.json"
COLLAB_EMB="./data/${DATASET}/${DATASET}.emb-collab-256.npy"

# CPA 配置 (Instruments 最佳)
CPA_W=0.005
CPA_LOSS=cosine

# 序列增强配置 (Instruments aug2 最佳)
AUG_DROPOUT=0.1
AUG_CROP=0.3

# beam 配置
NUM_BEAMS=50

OUTPUT_DIR="/sda/data/yaoxianglin/MACRec/log/${DATASET}-E1-lcpa${CPA_W}-aug-b${NUM_BEAMS}"

cd /sda/data/yaoxianglin/MACRec

echo "============================================"
echo " Inst E1 + LCPA w=${CPA_W} + Aug (dp=${AUG_DROPOUT},crop=${AUG_CROP}) + beam-${NUM_BEAMS}"
echo " GPU: ${TRAIN_GPUS} | Port: ${PORT}"
echo " 索引: ${INDEX_FILE} / ${IMAGE_INDEX_FILE}"
echo " 协同嵌入: ${COLLAB_EMB}"
echo " 输出: ${OUTPUT_DIR}"
echo " 开始时间: $(date)"
echo "============================================"

# ================================================================
# Step 1: T5 训练
# ================================================================
echo "[Step 1] T5 训练 (LCPA + Aug)... $(date)"
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
    --tasks seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image \
    --valid_task seqrec \
    --cpa_weight $CPA_W \
    --collab_emb_path $COLLAB_EMB \
    --cpa_loss_type $CPA_LOSS \
    --aug_item_dropout $AUG_DROPOUT \
    --aug_crop_prob $AUG_CROP 2>&1 | tee $OUTPUT_DIR/train.log

if [ ! -f "$OUTPUT_DIR/model.safetensors" ] && [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
    echo "[Step 1] 训练失败！模型文件未保存"
    exit 1
fi
echo "[Step 1] 训练完成 $(date)"

# ================================================================
# Step 2: seqrec 推理 (beam-50)
# ================================================================
echo "[Step 2] seqrec 推理 (beam-${NUM_BEAMS})... $(date)"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json \
    --filter_items

echo "[Step 2] 完成 $(date)"

# ================================================================
# Step 3: seqimage 推理 (beam-50)
# ================================================================
echo "[Step 3] seqimage 推理 (beam-${NUM_BEAMS})... $(date)"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json \
    --filter_items

echo "[Step 3] 完成 $(date)"

# ================================================================
# Step 4: Ensemble
# ================================================================
echo "[Step 4] Ensemble (beam-${NUM_BEAMS})... $(date)"

python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams $NUM_BEAMS

echo "[Step 4] 完成 $(date)"

# ================================================================
echo ""
echo "============================================"
echo " 全流程完成！$(date)"
echo " 配置: E1 + LCPA w=${CPA_W} ${CPA_LOSS} + Aug dp=${AUG_DROPOUT} crop=${AUG_CROP} + beam-${NUM_BEAMS}"
echo "============================================"
for f in results_seqrec_${NUM_BEAMS}.json results_seqimage_${NUM_BEAMS}.json results_ensemble_${NUM_BEAMS}.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
