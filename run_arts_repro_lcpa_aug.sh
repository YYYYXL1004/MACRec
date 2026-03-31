#!/bin/bash
# Arts-repro RQVAE + 轻量CPA + 序列增强
# 参照 Instruments 最佳配置: LCPA w=0.005 cosine + aug dropout=0.1 crop=0.3
# RQVAE索引: Arts-repro (LLaMA, 无collab, 有CAQ)
# GPU: 3,4 双卡

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Arts"
SAVE_NAME="Arts-repro"
TRAIN_GPUS="1,2"
PORT=29651

INDEX_FILE=".index_lemb_${SAVE_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${SAVE_NAME}.json"
COLLAB_EMB="./data/${DATASET}/${DATASET}.emb-collab-256.npy"

# CPA 配置 (Instruments 最佳)
CPA_W=0.005
CPA_LOSS=cosine

# 序列增强配置 (Instruments aug2 最佳)
AUG_DROPOUT=0.1
AUG_CROP=0.3

OUTPUT_DIR="/sda/data/yaoxianglin/MACRec/log/${DATASET}-${SAVE_NAME}-lcpa${CPA_W}-aug"

cd /sda/data/yaoxianglin/MACRec

echo "============================================"
echo " Arts-repro + LCPA w=${CPA_W} + Aug (dp=${AUG_DROPOUT},crop=${AUG_CROP})"
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
# Step 2: seqrec 推理
# ================================================================
echo "[Step 2] seqrec 推理... $(date)"

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

echo "[Step 2] 完成 $(date)"

# ================================================================
# Step 3: seqimage 推理
# ================================================================
echo "[Step 3] seqimage 推理... $(date)"

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

echo "[Step 3] 完成 $(date)"

# ================================================================
# Step 4: Ensemble
# ================================================================
echo "[Step 4] Ensemble... $(date)"

python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams 20

echo "[Step 4] 完成 $(date)"

# ================================================================
echo ""
echo "============================================"
echo " 全流程完成！$(date)"
echo " 配置: LCPA w=${CPA_W} ${CPA_LOSS} + Aug dp=${AUG_DROPOUT} crop=${AUG_CROP}"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
