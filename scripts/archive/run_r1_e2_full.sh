#!/bin/bash
# Round 1 E2 全流程: ST + concat collab, 无CAQ
# GPU 2,3 双卡 DDP (与 E6 共用，每卡约 14GB，共 28GB / 32GB)
# 用法: screen -dmS r1_e2 bash run_r1_e2_full.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_NAME="R1_E2_st-concat-nocaq"
TRAIN_GPUS="2,3"
PORT=29611

DATA_DIR="/data/yaoxianglin/ETEGRec/MACRec/data/${DATASET}"
RQVAE_DIR="/data/yaoxianglin/ETEGRec/MACRec/cross_index/log/${DATASET}/${EXP_NAME}"
OUTPUT_DIR="/data/yaoxianglin/ETEGRec/MACRec/log/${DATASET}-${EXP_NAME}"

TEXT_ST="${DATA_DIR}/${DATASET}.emb-st-768.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"

INDEX_FILE=".index_lemb_${EXP_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"

CHECKPOINT_DIR="/data/yaoxianglin/ETEGRec/MACRec/.pipeline_r1_e2"
mkdir -p $CHECKPOINT_DIR

cd /data/yaoxianglin/ETEGRec/MACRec

echo "============================================"
echo " E2 全流程: ${EXP_NAME}"
echo " 开始时间: $(date)"
echo "============================================"

# ================================================================
# Step 1: 生成 text index code
# ================================================================
if [ -f "$CHECKPOINT_DIR/step1_done" ]; then
    echo "[Step 1] text code 已生成，跳过"
else
    echo "[Step 1] 生成 text index code... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index
    python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --device cuda:2 \
        --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${DATASET}${INDEX_FILE} \
        --content text
    if [ -f "${DATA_DIR}/${DATASET}${INDEX_FILE}" ]; then
        touch "$CHECKPOINT_DIR/step1_done"
        echo "[Step 1] 完成 $(date)"
    else
        echo "[Step 1] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 2: 生成 image index code
# ================================================================
if [ -f "$CHECKPOINT_DIR/step2_done" ]; then
    echo "[Step 2] image code 已生成，跳过"
else
    echo "[Step 2] 生成 image index code... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index
    python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --device cuda:2 \
        --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${DATASET}${IMAGE_INDEX_FILE} \
        --content image
    if [ -f "${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}" ]; then
        touch "$CHECKPOINT_DIR/step2_done"
        echo "[Step 2] 完成 $(date)"
    else
        echo "[Step 2] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 3: T5 训练
# ================================================================
if [ -f "$CHECKPOINT_DIR/step3_done" ]; then
    echo "[Step 3] T5 训练已完成，跳过"
else
    echo "[Step 3] T5 训练... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec
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
        --valid_task seqrec 2>&1 | tee $OUTPUT_DIR/train.log

    if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]; then
        touch "$CHECKPOINT_DIR/step3_done"
        echo "[Step 3] 完成 $(date)"
    else
        echo "[Step 3] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 4: 推理 seqrec
# ================================================================
if [ -f "$CHECKPOINT_DIR/step4_done" ]; then
    echo "[Step 4] seqrec 推理已完成，跳过"
else
    echo "[Step 4] seqrec 推理... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec

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

    if [ -f "$OUTPUT_DIR/results_seqrec_20.json" ]; then
        touch "$CHECKPOINT_DIR/step4_done"
        echo "[Step 4] 完成 $(date)"
    else
        echo "[Step 4] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 5: 推理 seqimage
# ================================================================
if [ -f "$CHECKPOINT_DIR/step5_done" ]; then
    echo "[Step 5] seqimage 推理已完成，跳过"
else
    echo "[Step 5] seqimage 推理... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec

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

    if [ -f "$OUTPUT_DIR/results_seqimage_20.json" ]; then
        touch "$CHECKPOINT_DIR/step5_done"
        echo "[Step 5] 完成 $(date)"
    else
        echo "[Step 5] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 6: Ensemble
# ================================================================
if [ -f "$CHECKPOINT_DIR/step6_done" ]; then
    echo "[Step 6] ensemble 已完成，跳过"
else
    echo "[Step 6] Ensemble... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec

    python ensemble.py \
        --output_dir $OUTPUT_DIR \
        --dataset $DATASET \
        --data_path ./data/ \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --num_beams 20

    if [ -f "$OUTPUT_DIR/results_ensemble_20.json" ]; then
        touch "$CHECKPOINT_DIR/step6_done"
        echo "[Step 6] 完成 $(date)"
    else
        echo "[Step 6] 失败！"; exit 1
    fi
fi

echo ""
echo "============================================"
echo " E2 全流程完成！$(date)"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
