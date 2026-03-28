#!/bin/bash
# Arts 全流程: LLaMA+concat+CAQ (协同锚定)
# GPU 2,3 双卡 (RQVAE单卡用2, T5双卡用2,3)
# 用法: screen -dmS arts_llama bash run_arts_llama_concat_caq.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Arts"
EXP_NAME="R2_llama-concat-caq"
RQVAE_VIS_GPU="2"
TRAIN_GPUS="2,3"
PORT=29611

DATA_DIR="/data/yaoxianglin/ETEGRec/MACRec/data/${DATASET}"
RQVAE_DIR="/data/yaoxianglin/ETEGRec/MACRec/cross_index/log/${DATASET}/${EXP_NAME}"
OUTPUT_DIR="/data/yaoxianglin/ETEGRec/MACRec/log/${DATASET}-${EXP_NAME}"

# LLaMA text + ViT image + collab
TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

INDEX_FILE=".index_lemb_${EXP_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"

CHECKPOINT_DIR="/data/yaoxianglin/ETEGRec/MACRec/.pipeline_arts_llama_concat"
mkdir -p $CHECKPOINT_DIR

cd /data/yaoxianglin/ETEGRec/MACRec

echo "============================================"
echo " Arts LLaMA+concat+CAQ 全流程"
echo " RQVAE: GPU ${RQVAE_VIS_GPU} | T5: GPU ${TRAIN_GPUS}"
echo " 开始时间: $(date)"
echo "============================================"

# ================================================================
# Step 1: 训练 RQVAE (LLaMA+concat+CAQ)
# ================================================================
if [ -f "$CHECKPOINT_DIR/step1_done" ]; then
    echo "[Step 1] RQVAE 已完成，跳过"
else
    echo "[Step 1] 训练 RQVAE (LLaMA+concat+CAQ)... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index
    mkdir -p $RQVAE_DIR

    CUDA_VISIBLE_DEVICES=${RQVAE_VIS_GPU} python -u main.py \
        --device cuda:0 \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} \
        --collab_fusion concat \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight 2.0 \
        --ckpt_dir $RQVAE_DIR \
        --num_emb_list 256 256 256 256 \
        --sk_epsilons 0.0 0.0 0.0 0.0 \
        --eval_step 2 \
        --batch_size 2048 \
        --begin_cross_layer 2 \
        --use_cross_rq True \
        --text_class_info ${TEXT_CLASS} \
        --image_class_info ${IMAGE_CLASS} \
        --text_contrast_weight 0.1 \
        --image_contrast_weight 0.1 \
        --recon_contrast_weight 0.001 \
        --epochs 1000 2>&1 | tee $RQVAE_DIR/train.log

    if [ -f "$RQVAE_DIR/best_text_collision_model.pth" ] && [ -f "$RQVAE_DIR/best_image_collision_model.pth" ]; then
        touch "$CHECKPOINT_DIR/step1_done"
        echo "[Step 1] 完成 $(date)"
    else
        echo "[Step 1] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 2: 生成 text index code
# ================================================================
if [ -f "$CHECKPOINT_DIR/step2_done" ]; then
    echo "[Step 2] text code 已生成，跳过"
else
    echo "[Step 2] 生成 text index code... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index

    CUDA_VISIBLE_DEVICES=${RQVAE_VIS_GPU} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${DATASET}${INDEX_FILE} \
        --content text

    if [ -f "${DATA_DIR}/${DATASET}${INDEX_FILE}" ]; then
        touch "$CHECKPOINT_DIR/step2_done"
        echo "[Step 2] 完成 $(date)"
    else
        echo "[Step 2] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 3: 生成 image index code
# ================================================================
if [ -f "$CHECKPOINT_DIR/step3_done" ]; then
    echo "[Step 3] image code 已生成，跳过"
else
    echo "[Step 3] 生成 image index code... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index

    CUDA_VISIBLE_DEVICES=${RQVAE_VIS_GPU} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${DATASET}${IMAGE_INDEX_FILE} \
        --content image

    if [ -f "${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}" ]; then
        touch "$CHECKPOINT_DIR/step3_done"
        echo "[Step 3] 完成 $(date)"
    else
        echo "[Step 3] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 4: T5 训练
# ================================================================
if [ -f "$CHECKPOINT_DIR/step4_done" ]; then
    echo "[Step 4] T5 训练已完成，跳过"
else
    echo "[Step 4] T5 训练... $(date)"
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
        touch "$CHECKPOINT_DIR/step4_done"
        echo "[Step 4] 完成 $(date)"
    else
        echo "[Step 4] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 5: 推理 seqrec
# ================================================================
if [ -f "$CHECKPOINT_DIR/step5_done" ]; then
    echo "[Step 5] seqrec 推理已完成，跳过"
else
    echo "[Step 5] seqrec 推理... $(date)"
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
        touch "$CHECKPOINT_DIR/step5_done"
        echo "[Step 5] 完成 $(date)"
    else
        echo "[Step 5] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 6: 推理 seqimage
# ================================================================
if [ -f "$CHECKPOINT_DIR/step6_done" ]; then
    echo "[Step 6] seqimage 推理已完成，跳过"
else
    echo "[Step 6] seqimage 推理... $(date)"
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
        touch "$CHECKPOINT_DIR/step6_done"
        echo "[Step 6] 完成 $(date)"
    else
        echo "[Step 6] 失败！"; exit 1
    fi
fi

# ================================================================
# Step 7: Ensemble
# ================================================================
if [ -f "$CHECKPOINT_DIR/step7_done" ]; then
    echo "[Step 7] ensemble 已完成，跳过"
else
    echo "[Step 7] Ensemble... $(date)"
    cd /data/yaoxianglin/ETEGRec/MACRec

    python ensemble.py \
        --output_dir $OUTPUT_DIR \
        --dataset $DATASET \
        --data_path ./data/ \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --num_beams 20

    if [ -f "$OUTPUT_DIR/results_ensemble_20.json" ]; then
        touch "$CHECKPOINT_DIR/step7_done"
        echo "[Step 7] 完成 $(date)"
    else
        echo "[Step 7] 失败！"; exit 1
    fi
fi

echo ""
echo "============================================"
echo " Arts LLaMA+concat+CAQ 全流程完成！$(date)"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
