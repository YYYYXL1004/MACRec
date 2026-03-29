#!/bin/bash
# MACRec Games 全流程复现脚本 (带断点续跑 + 等待 GPU 3,4 释放)

eval "$(conda shell.bash hook)"
conda activate ETEGRec

DATASET="Games"
SAVE_NAME="Games-repro"
RQVAE_GPU="cuda:0"   # CUDA_VISIBLE_DEVICES 重映射后的 0
TRAIN_GPUS="4,5"
PORT=29700

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CHECKPOINT_DIR="/data/yaoxianglin/ETEGRec/MACRec/.pipeline_checkpoints_games"
mkdir -p $CHECKPOINT_DIR

RQVAE_DIR="/data/yaoxianglin/ETEGRec/MACRec/cross_index/log/${DATASET}/${SAVE_NAME}"
OUTPUT_DIR="/data/yaoxianglin/ETEGRec/MACRec/log/${DATASET}-${SAVE_NAME}-contrastive-align-0.01-temp-0.07-full-second"
INDEX_FILE=".index_lemb_${SAVE_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${SAVE_NAME}.json"

echo "============================================"
echo " MACRec Games 全流程复现"
echo " RQVAE: GPU 4 (单卡)"
echo " T5: GPU 4,5 (双卡)"
echo " 启动时间: $(date)"
echo "============================================"

# ================================================================
# Step 2b: 训练 CrossRQVAE
# ================================================================
if [ -f "$CHECKPOINT_DIR/step2b_done" ]; then
    echo "[Step 2b] CrossRQVAE 已完成，跳过"
else
    echo ""
    echo "[Step 2b] 训练 CrossRQVAE... $(date)"

    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index
    mkdir -p $RQVAE_DIR

    CUDA_VISIBLE_DEVICES=4 python -u main.py \
        --num_emb_list 256 256 256 256 \
        --sk_epsilons 0.0 0.0 0.0 0.0 \
        --device cuda:0 \
        --text_data_path ../data/${DATASET}/${DATASET}.emb-llama-td.npy \
        --image_data_path ../data/${DATASET}/${DATASET}.emb-ViT-L-14.npy \
        --ckpt_dir $RQVAE_DIR \
        --eval_step 2 \
        --batch_size 2048 \
        --begin_cross_layer 2 \
        --use_cross_rq True \
        --text_class_info ../data/${DATASET}/${DATASET}.index_lemb_kmeans512.json \
        --image_class_info ../data/${DATASET}/${DATASET}.index_vitemb_kmeans512.json \
        --text_contrast_weight 0.1 \
        --image_contrast_weight 0.1 \
        --recon_contrast_weight 0.001 \
        --epochs 1000 2>&1 | tee $RQVAE_DIR/train.log

    if [ -f "$RQVAE_DIR/best_text_collision_model.pth" ] && [ -f "$RQVAE_DIR/best_image_collision_model.pth" ]; then
        touch "$CHECKPOINT_DIR/step2b_done"
        echo "[Step 2b] 完成 $(date)"
    else
        echo "[Step 2b] 失败！缺少 best model 文件"
        exit 1
    fi
fi

# ================================================================
# Step 2c: 生成离散索引码
# ================================================================
if [ -f "$CHECKPOINT_DIR/step2c_done" ]; then
    echo "[Step 2c] 索引生成已完成，跳过"
else
    echo ""
    echo "[Step 2c] 生成离散索引码... $(date)"

    cd /data/yaoxianglin/ETEGRec/MACRec/cross_index

    CUDA_VISIBLE_DEVICES=4 python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ../data/${DATASET}/${DATASET}.emb-llama-td.npy \
        --image_data_path ../data/${DATASET}/${DATASET}.emb-ViT-L-14.npy \
        --device cuda:0 \
        --ckpt_path $RQVAE_DIR/best_text_collision_model.pth \
        --output_dir ../data/${DATASET} \
        --output_file ${DATASET}.index_lemb_${SAVE_NAME}.json \
        --content text

    CUDA_VISIBLE_DEVICES=4 python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ../data/${DATASET}/${DATASET}.emb-llama-td.npy \
        --image_data_path ../data/${DATASET}/${DATASET}.emb-ViT-L-14.npy \
        --device cuda:0 \
        --ckpt_path $RQVAE_DIR/best_image_collision_model.pth \
        --output_dir ../data/${DATASET} \
        --output_file ${DATASET}.index_vitemb_${SAVE_NAME}.json \
        --content image

    TEXT_IDX="../data/${DATASET}/${DATASET}.index_lemb_${SAVE_NAME}.json"
    IMAGE_IDX="../data/${DATASET}/${DATASET}.index_vitemb_${SAVE_NAME}.json"
    if [ -f "$TEXT_IDX" ] && [ -f "$IMAGE_IDX" ]; then
        touch "$CHECKPOINT_DIR/step2c_done"
        echo "[Step 2c] 完成 $(date)"
    else
        echo "[Step 2c] 失败！索引文件未生成"
        exit 1
    fi
fi

# ================================================================
# Step 3: T5 训练
# ================================================================
if [ -f "$CHECKPOINT_DIR/step3_done" ]; then
    echo "[Step 3] T5 训练已完成，跳过"
else
    echo ""
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
        echo "[Step 3] 失败！模型文件未保存"
        exit 1
    fi
fi

# ================================================================
# Step 4a: 文本路推理
# ================================================================
if [ -f "$CHECKPOINT_DIR/step4a_done" ]; then
    echo "[Step 4a] 文本路推理已完成，跳过"
else
    echo ""
    echo "[Step 4a] 文本路推理... $(date)"

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
        touch "$CHECKPOINT_DIR/step4a_done"
        echo "[Step 4a] 完成 $(date)"
    else
        echo "[Step 4a] 失败！"
        exit 1
    fi
fi

# ================================================================
# Step 4b: 图像路推理
# ================================================================
if [ -f "$CHECKPOINT_DIR/step4b_done" ]; then
    echo "[Step 4b] 图像路推理已完成，跳过"
else
    echo ""
    echo "[Step 4b] 图像路推理... $(date)"

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
        touch "$CHECKPOINT_DIR/step4b_done"
        echo "[Step 4b] 完成 $(date)"
    else
        echo "[Step 4b] 失败！"
        exit 1
    fi
fi

# ================================================================
# Step 4c: Ensemble
# ================================================================
if [ -f "$CHECKPOINT_DIR/step4c_done" ]; then
    echo "[Step 4c] Ensemble 已完成，跳过"
else
    echo ""
    echo "[Step 4c] 双路 Ensemble... $(date)"

    cd /data/yaoxianglin/ETEGRec/MACRec

    python ensemble.py \
        --output_dir $OUTPUT_DIR \
        --dataset $DATASET \
        --data_path ./data/ \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --num_beams 20

    if [ -f "$OUTPUT_DIR/results_ensemble_20.json" ]; then
        touch "$CHECKPOINT_DIR/step4c_done"
        echo "[Step 4c] 完成 $(date)"
    else
        echo "[Step 4c] 失败！"
        exit 1
    fi
fi

# ================================================================
echo ""
echo "============================================"
echo " Games 全流程完成！结束时间: $(date)"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
