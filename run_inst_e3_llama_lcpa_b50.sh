#!/bin/bash
# Instruments LLaMA+concat+CAQ 全流程实验 (从RQVAE训练开始)
# 实验1: LLaMA+concat+CAQ+LCPA+beam-50 (无aug)
# 实验2: LLaMA+concat+CAQ+LCPA+aug+beam-50
# GPU: 0,1 (RQVAE用GPU 0单卡, T5用GPU 0,1双卡)

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_NAME="E3v2_llama-concat-caq"
RQVAE_GPU="0"
TRAIN_GPUS="0,1"
PORT=29660

BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"
RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_NAME}"

# 数据文件
TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

# 索引文件名
INDEX_FILE=".index_lemb_${EXP_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'

# CPA 配置
CPA_W=0.005
CPA_LOSS=cosine

# beam 配置
NUM_BEAMS=50

cd $BASE_DIR

echo "============================================"
echo " Inst LLaMA+concat+CAQ 全流程 (RQVAE→T5→推理)"
echo " RQVAE: GPU ${RQVAE_GPU} | T5: GPU ${TRAIN_GPUS}"
echo " 开始时间: $(date)"
echo "============================================"

mkdir -p $RQVAE_DIR

# ################################################################
# Phase 0: RQVAE 训练 (单卡)
# 配置: LLaMA(4096d)+collab(256d) concat, 256×4 codebook, e_dim=32, CAQ w=2.0
# ################################################################
echo ""
echo "================================================================"
echo " [Phase 0] RQVAE 训练 (LLaMA+concat+CAQ, 256×4, e_dim=32)"
echo " GPU: ${RQVAE_GPU} | 输出: ${RQVAE_DIR}"
echo " 开始: $(date)"
echo "================================================================"

cd ${BASE_DIR}/cross_index

CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} \
    --image_data_path ${IMAGE} \
    --collab_data_path ${COLLAB} \
    --collab_fusion concat \
    --collab_neighbor_info ${COLLAB_NEIGHBOR} \
    --collab_contrastive_weight 2.0 \
    --ckpt_dir $RQVAE_DIR \
    --num_emb_list 256 256 256 256 \
    --e_dim 32 \
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

if [ ! -f "$RQVAE_DIR/best_text_collision_model.pth" ] || [ ! -f "$RQVAE_DIR/best_image_collision_model.pth" ]; then
    echo "[Phase 0] RQVAE 训练失败！模型文件未保存"
    exit 1
fi
echo "[Phase 0] RQVAE 训练完成 $(date)"

# ################################################################
# Phase 1: 生成离散索引 (text + image)
# ################################################################
echo ""
echo "================================================================"
echo " [Phase 1] 生成离散索引"
echo "================================================================"

echo "[Phase 1a] 生成 text code... $(date)"
CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
    --dataset $DATASET \
    --text_data_path ${TEXT_LLAMA} \
    --image_data_path ${IMAGE} \
    --device cuda:0 \
    --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
    --output_dir ${DATA_DIR} \
    --output_file ${DATASET}${INDEX_FILE} \
    --content text

echo "[Phase 1b] 生成 image code... $(date)"
CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
    --dataset $DATASET \
    --text_data_path ${TEXT_LLAMA} \
    --image_data_path ${IMAGE} \
    --device cuda:0 \
    --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
    --output_dir ${DATA_DIR} \
    --output_file ${DATASET}${IMAGE_INDEX_FILE} \
    --content image

# 验证索引文件
if [ ! -f "${DATA_DIR}/${DATASET}${INDEX_FILE}" ] || [ ! -f "${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}" ]; then
    echo "[Phase 1] 索引生成失败！"
    exit 1
fi
echo "[Phase 1] 索引生成完成 $(date)"

# 打印碰撞率
python -c "
import json
for name, path in [('text', '${DATA_DIR}/${DATASET}${INDEX_FILE}'), ('image', '${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    unique = len(set(codes))
    total = len(codes)
    collision = (1 - unique/total) * 100
    print(f'  {name}: {total} items, {unique} unique codes, collision={collision:.2f}%')
"

cd $BASE_DIR

# ################################################################
# 实验1: LLaMA+concat+CAQ+LCPA+beam-50 (无aug)
# ################################################################
OUTPUT_DIR1="${BASE_DIR}/log/${DATASET}-${EXP_NAME}-lcpa${CPA_W}-b${NUM_BEAMS}"

echo ""
echo "============================================"
echo " [实验1] LCPA w=${CPA_W} + beam-${NUM_BEAMS} (无aug)"
echo " GPU: ${TRAIN_GPUS} | Port: ${PORT}"
echo " 索引: ${INDEX_FILE} / ${IMAGE_INDEX_FILE}"
echo " 输出: ${OUTPUT_DIR1}"
echo " 开始: $(date)"
echo "============================================"

# Step 1: T5 训练 (LCPA only, 无aug)
echo "[实验1-Step1] T5 训练 (LCPA only)... $(date)"
mkdir -p $OUTPUT_DIR1

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR1 \
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
    --valid_task seqrec \
    --cpa_weight $CPA_W \
    --collab_emb_path $COLLAB \
    --cpa_loss_type $CPA_LOSS 2>&1 | tee $OUTPUT_DIR1/train.log

if [ ! -f "$OUTPUT_DIR1/model.safetensors" ] && [ ! -f "$OUTPUT_DIR1/pytorch_model.bin" ]; then
    echo "[实验1-Step1] 训练失败！"
    exit 1
fi
echo "[实验1-Step1] 训练完成 $(date)"

# Step 2: seqrec 推理 (beam-50)
echo "[实验1-Step2] seqrec 推理 (beam-${NUM_BEAMS})... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR1 \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR1/results_seqrec_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR1/save_seqrec_${NUM_BEAMS}.json \
    --filter_items
echo "[实验1-Step2] 完成 $(date)"

# Step 3: seqimage 推理 (beam-50)
echo "[实验1-Step3] seqimage 推理 (beam-${NUM_BEAMS})... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR1 \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR1/results_seqimage_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR1/save_seqimage_${NUM_BEAMS}.json \
    --filter_items
echo "[实验1-Step3] 完成 $(date)"

# Step 4: Ensemble
echo "[实验1-Step4] Ensemble (beam-${NUM_BEAMS})... $(date)"
python ensemble.py \
    --output_dir $OUTPUT_DIR1 \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams $NUM_BEAMS
echo "[实验1-Step4] 完成 $(date)"

echo ""
echo "============================================"
echo " [实验1] 全流程完成！$(date)"
echo " 配置: LLaMA+concat+CAQ + LCPA w=${CPA_W} + beam-${NUM_BEAMS}"
echo "============================================"
for f in results_seqrec_${NUM_BEAMS}.json results_seqimage_${NUM_BEAMS}.json results_ensemble_${NUM_BEAMS}.json; do
    if [ -f "$OUTPUT_DIR1/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR1/$f"
        echo ""
    fi
done

# ################################################################
# 实验2: LLaMA+concat+CAQ+LCPA+aug+beam-50
# ################################################################
AUG_DROPOUT=0.1
AUG_CROP=0.3
OUTPUT_DIR2="${BASE_DIR}/log/${DATASET}-${EXP_NAME}-lcpa${CPA_W}-aug-b${NUM_BEAMS}"

echo ""
echo "============================================"
echo " [实验2] LCPA w=${CPA_W} + Aug (dp=${AUG_DROPOUT},crop=${AUG_CROP}) + beam-${NUM_BEAMS}"
echo " GPU: ${TRAIN_GPUS} | Port: ${PORT}"
echo " 输出: ${OUTPUT_DIR2}"
echo " 开始: $(date)"
echo "============================================"

# Step 1: T5 训练 (LCPA + Aug)
echo "[实验2-Step1] T5 训练 (LCPA + Aug)... $(date)"
mkdir -p $OUTPUT_DIR2

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR2 \
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
    --valid_task seqrec \
    --cpa_weight $CPA_W \
    --collab_emb_path $COLLAB \
    --cpa_loss_type $CPA_LOSS \
    --aug_item_dropout $AUG_DROPOUT \
    --aug_crop_prob $AUG_CROP 2>&1 | tee $OUTPUT_DIR2/train.log

if [ ! -f "$OUTPUT_DIR2/model.safetensors" ] && [ ! -f "$OUTPUT_DIR2/pytorch_model.bin" ]; then
    echo "[实验2-Step1] 训练失败！"
    exit 1
fi
echo "[实验2-Step1] 训练完成 $(date)"

# Step 2: seqrec 推理 (beam-50)
echo "[实验2-Step2] seqrec 推理 (beam-${NUM_BEAMS})... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR2 \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR2/results_seqrec_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR2/save_seqrec_${NUM_BEAMS}.json \
    --filter_items
echo "[实验2-Step2] 完成 $(date)"

# Step 3: seqimage 推理 (beam-50)
echo "[实验2-Step3] seqimage 推理 (beam-${NUM_BEAMS})... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR2 \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR2/results_seqimage_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR2/save_seqimage_${NUM_BEAMS}.json \
    --filter_items
echo "[实验2-Step3] 完成 $(date)"

# Step 4: Ensemble
echo "[实验2-Step4] Ensemble (beam-${NUM_BEAMS})... $(date)"
python ensemble.py \
    --output_dir $OUTPUT_DIR2 \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams $NUM_BEAMS
echo "[实验2-Step4] 完成 $(date)"

echo ""
echo "============================================"
echo " [实验2] 全流程完成！$(date)"
echo " 配置: LLaMA+concat+CAQ + LCPA w=${CPA_W} + Aug dp=${AUG_DROPOUT} crop=${AUG_CROP} + beam-${NUM_BEAMS}"
echo "============================================"
for f in results_seqrec_${NUM_BEAMS}.json results_seqimage_${NUM_BEAMS}.json results_ensemble_${NUM_BEAMS}.json; do
    if [ -f "$OUTPUT_DIR2/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR2/$f"
        echo ""
    fi
done

echo ""
echo "========================================================"
echo " 两组实验全部完成！$(date)"
echo " RQVAE: ${RQVAE_DIR}"
echo " 实验1: ${OUTPUT_DIR1} (LCPA only)"
echo " 实验2: ${OUTPUT_DIR2} (LCPA + Aug)"
echo "========================================================"
