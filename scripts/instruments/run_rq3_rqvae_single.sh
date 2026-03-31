#!/bin/bash
# RQ3: 单个 fusion 类型的 RQVAE 训练 + 索引生成
# 用法: bash run_rq3_rqvae_single.sh <FUSION_TYPE> <GPU_ID>
# 例如: bash run_rq3_rqvae_single.sh proj_add 2

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled

FUSION_TYPE="${1:?需要指定 fusion 类型: proj_add / gating / cross_attn}"
GPU_ID="${2:?需要指定 GPU ID}"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"
CAR_W=2.0

EXP_TAG="rq3-${FUSION_TYPE}"
RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_TAG}"
INDEX_FILE=".index_lemb_${EXP_TAG}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_TAG}.json"

echo "================================================================"
echo " RQ3 RQVAE: fusion=${FUSION_TYPE}, GPU=${GPU_ID}"
echo " 输出目录: ${RQVAE_DIR}"
echo " 开始: $(date)"
echo "================================================================"

# ============ Phase 0: RQVAE 训练 ============
mkdir -p $RQVAE_DIR
cd ${BASE_DIR}/cross_index

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
    --collab_data_path ${COLLAB} --collab_fusion ${FUSION_TYPE} \
    --collab_neighbor_info ${COLLAB_NEIGHBOR} --collab_contrastive_weight ${CAR_W} \
    --ckpt_dir $RQVAE_DIR \
    --num_emb_list 256 256 256 256 --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --eval_step 2 --batch_size 2048 \
    --begin_cross_layer 2 --use_cross_rq True \
    --text_class_info ${TEXT_CLASS} --image_class_info ${IMAGE_CLASS} \
    --text_contrast_weight 0.1 --image_contrast_weight 0.1 \
    --recon_contrast_weight 0.001 \
    --epochs 1000 2>&1 | tee $RQVAE_DIR/train.log

if [ ! -f "$RQVAE_DIR/best_text_collision_model.pth" ]; then
    echo " [${EXP_TAG}] RQVAE 训练失败!"
    exit 1
fi

echo " [${EXP_TAG}] RQVAE 训练完成，开始生成索引..."

# ============ Phase 1: 索引生成 ============
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u generate_indices_distance.py \
    --dataset $DATASET --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
    --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
    --output_dir ${DATA_DIR} --output_file ${DATASET}${INDEX_FILE} --content text

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u generate_indices_distance.py \
    --dataset $DATASET --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
    --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
    --output_dir ${DATA_DIR} --output_file ${DATASET}${IMAGE_INDEX_FILE} --content image

# 碰撞率
python -c "
import json
for name, path in [('text', '${DATA_DIR}/${DATASET}${INDEX_FILE}'), ('image', '${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    print(f'  {name}: {len(codes)} items, {len(set(codes))} unique, collision={(1-len(set(codes))/len(codes))*100:.2f}%')
"

echo ""
echo "================================================================"
echo " [${EXP_TAG}] RQVAE + 索引生成完成! $(date)"
echo "================================================================"
