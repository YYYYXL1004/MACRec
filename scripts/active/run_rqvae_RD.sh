#!/bin/bash
# RQVAE R-D: 256×4 + e_dim=128（大宽度消融, encoder去掉末端64层）
# RQVAE 单卡 GPU 1, T5 双卡 GPU 0,1 (与 R-C 共享)
# 用法: screen -dmS rq_rd bash scripts/active/run_rqvae_RD.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_NAME="RQ-RD_256x4-ed128"
RQVAE_GPU="1"
TRAIN_GPUS="0,1"
PORT=29722

DATA_DIR="/data/yaoxianglin/MACRec/data/${DATASET}"
RQVAE_DIR="/data/yaoxianglin/MACRec/cross_index/log/${DATASET}/${EXP_NAME}"
OUTPUT_DIR="/data/yaoxianglin/MACRec/log/${DATASET}-${EXP_NAME}"

TEXT_ST="${DATA_DIR}/${DATASET}.emb-st-768.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

INDEX_FILE=".index_lemb_${EXP_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'

cd /data/yaoxianglin/MACRec

echo "============================================"
echo " RQVAE R-D: 256×4 + e_dim=128 (encoder去掉末端64层)"
echo " RQVAE: GPU ${RQVAE_GPU} | T5: GPU ${TRAIN_GPUS}"
echo " 开始时间: $(date)"
echo "============================================"

mkdir -p $RQVAE_DIR $OUTPUT_DIR

# Step 1: RQVAE 训练（单卡）
# encoder 层: [align_dim=768, 2048, 1024, 512, 256, 128, e_dim=128]
# 去掉默认的末端 64，使最终压缩为 256→128 而非 64→128 (扩张)
echo "[R-D][Step 1] RQVAE 训练... $(date)"
cd /data/yaoxianglin/MACRec/cross_index

CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_ST} \
    --image_data_path ${IMAGE} \
    --collab_data_path ${COLLAB} \
    --collab_fusion concat \
    --collab_neighbor_info ${COLLAB_NEIGHBOR} \
    --collab_contrastive_weight 2.0 \
    --ckpt_dir $RQVAE_DIR \
    --num_emb_list 256 256 256 256 \
    --e_dim 128 \
    --layers 2048 1024 512 256 128 \
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

echo "[R-D][Step 1] RQVAE 完成 $(date)"

# Step 2-3: 生成 code
echo "[R-D][Step 2] 生成 text code... $(date)"
CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
    --dataset $DATASET --text_data_path ${TEXT_ST} --image_data_path ${IMAGE} \
    --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
    --output_dir ${DATA_DIR} --output_file ${DATASET}${INDEX_FILE} --content text

echo "[R-D][Step 3] 生成 image code... $(date)"
CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
    --dataset $DATASET --text_data_path ${TEXT_ST} --image_data_path ${IMAGE} \
    --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
    --output_dir ${DATA_DIR} --output_file ${DATASET}${IMAGE_INDEX_FILE} --content image

# Step 4: T5 训练（双卡 DDP）
echo "[R-D][Step 4] T5 训练... $(date)"
cd /data/yaoxianglin/MACRec

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
    --base_model ./config/ckpt --per_device_batch_size 1024 --learning_rate 1e-3 \
    --epochs 200 --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
    --max_his_len 20 --prompt_num 4 --patient 10 \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --tasks $TASKS --valid_task seqrec 2>&1 | tee $OUTPUT_DIR/train.log

# Step 5-7: 推理 + Ensemble
echo "[R-D][Step 5] seqrec 推理... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET --test_batch_size 64 --num_beams 20 \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec --results_file $OUTPUT_DIR/results_seqrec_20.json --save_file $OUTPUT_DIR/save_seqrec_20.json --filter_items

echo "[R-D][Step 6] seqimage 推理... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET --test_batch_size 64 --num_beams 20 \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage --results_file $OUTPUT_DIR/results_seqimage_20.json --save_file $OUTPUT_DIR/save_seqimage_20.json --filter_items

echo "[R-D][Step 7] Ensemble... $(date)"
python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET --data_path ./data/ \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams 20

echo "============================================"
echo " R-D 全流程完成！$(date)"
echo "============================================"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    [ -f "$OUTPUT_DIR/$f" ] && echo "[$f]:" && cat "$OUTPUT_DIR/$f" && echo ""
done
