#!/bin/bash
# Games RQ2 消融: RQVAE 训练 + 索引生成
# 消融1 (w/o concat): collab_data_path 置空, 保留 CAQ
# 消融2 (w/o CAQ):    保留 concat, collab_contrastive_weight=0
# 自动寻找 2 张空闲 GPU (>8GB), 并行训练
# 用法: nohup bash scripts/games/run_games_abl_rqvae.sh > run_games_abl_rqvae.out 2>&1 &

eval "$(conda shell.bash hook)"
conda activate ETEGRec
export WANDB_MODE=disabled

DATASET="Games"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

FREE_MEM_THRESHOLD=8000  # RQVAE 单卡 ~4GB, 留余量

cd ${BASE_DIR}/cross_index

# ============ 自动寻找 2 张空闲 GPU ============
find_free_gpus() {
    local needed=$1
    local gpus=()
    while IFS=',' read -r idx mem_free; do
        idx=$(echo "$idx" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
        if [ "$mem_free" -ge "$FREE_MEM_THRESHOLD" ]; then
            gpus+=("$idx")
        fi
        [ "${#gpus[@]}" -ge "$needed" ] && break
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    if [ "${#gpus[@]}" -lt "$needed" ]; then
        echo ""
        return 1
    fi
    echo "${gpus[@]}"
}

GPU_LIST=$(find_free_gpus 2)
if [ -z "$GPU_LIST" ]; then
    echo "未找到 2 张空闲 GPU (>=${FREE_MEM_THRESHOLD}MB)，退出"
    exit 1
fi
GPU_ARR=($GPU_LIST)
GPU_WO_CONCAT=${GPU_ARR[0]}
GPU_WO_CAQ=${GPU_ARR[1]}

echo "=============================================="
echo " Games RQ2 消融 RQVAE 训练"
echo " w/o concat → GPU ${GPU_WO_CONCAT}"
echo " w/o CAQ    → GPU ${GPU_WO_CAQ}"
echo " 开始: $(date)"
echo "=============================================="

# ============ 消融1: w/o concat ============
# collab_data_path 置空 → 无 concat; 保留 CAQ + neighbor_info
WO_CONCAT_DIR="${BASE_DIR}/cross_index/log/${DATASET}/wo-concat_llama-caq"
mkdir -p $WO_CONCAT_DIR

CUDA_VISIBLE_DEVICES=${GPU_WO_CONCAT} python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
    --collab_data_path "" --collab_fusion concat \
    --collab_neighbor_info ${COLLAB_NEIGHBOR} --collab_contrastive_weight 2.0 \
    --ckpt_dir $WO_CONCAT_DIR \
    --num_emb_list 256 256 256 256 --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --eval_step 2 --batch_size 2048 \
    --begin_cross_layer 2 --use_cross_rq True \
    --text_class_info ${TEXT_CLASS} --image_class_info ${IMAGE_CLASS} \
    --text_contrast_weight 0.1 --image_contrast_weight 0.1 \
    --recon_contrast_weight 0.001 \
    --epochs 1000 2>&1 | tee $WO_CONCAT_DIR/train.log &
PID_WO_CONCAT=$!

# ============ 消融2: w/o CAQ ============
# 保留 concat; collab_contrastive_weight=0, neighbor_info 置空
WO_CAQ_DIR="${BASE_DIR}/cross_index/log/${DATASET}/wo-caq_llama-concat"
mkdir -p $WO_CAQ_DIR

CUDA_VISIBLE_DEVICES=${GPU_WO_CAQ} python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
    --collab_data_path ${COLLAB} --collab_fusion concat \
    --collab_neighbor_info "" --collab_contrastive_weight 0.0 \
    --ckpt_dir $WO_CAQ_DIR \
    --num_emb_list 256 256 256 256 --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --eval_step 2 --batch_size 2048 \
    --begin_cross_layer 2 --use_cross_rq True \
    --text_class_info ${TEXT_CLASS} --image_class_info ${IMAGE_CLASS} \
    --text_contrast_weight 0.1 --image_contrast_weight 0.1 \
    --recon_contrast_weight 0.001 \
    --epochs 1000 2>&1 | tee $WO_CAQ_DIR/train.log &
PID_WO_CAQ=$!

echo "RQVAE PID: w/o concat=${PID_WO_CONCAT} (GPU ${GPU_WO_CONCAT}), w/o CAQ=${PID_WO_CAQ} (GPU ${GPU_WO_CAQ})"
echo "等待两个 RQVAE 训练完成..."

wait $PID_WO_CONCAT
EXIT_WO_CONCAT=$?
echo "[$(date)] w/o concat RQVAE 完成 (exit=${EXIT_WO_CONCAT})"

wait $PID_WO_CAQ
EXIT_WO_CAQ=$?
echo "[$(date)] w/o CAQ RQVAE 完成 (exit=${EXIT_WO_CAQ})"

# ============ 验证模型 ============
for rdir in $WO_CONCAT_DIR $WO_CAQ_DIR; do
    if [ ! -f "$rdir/best_text_collision_model.pth" ] || [ ! -f "$rdir/best_image_collision_model.pth" ]; then
        echo "RQVAE 模型缺失: $rdir"
        exit 1
    fi
done

# ============ 索引生成 ============
echo ""
echo "================================================================"
echo " 生成离散索引 $(date)"
echo "================================================================"

generate_indices() {
    local EXP_NAME=$1
    local GPU=$2
    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_NAME}"
    local IDX_FILE="${DATASET}.index_lemb_${EXP_NAME}.json"
    local IMG_IDX_FILE="${DATASET}.index_vitemb_${EXP_NAME}.json"

    echo "  生成 ${EXP_NAME} text code (GPU ${GPU})..."
    CUDA_VISIBLE_DEVICES=${GPU} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${IDX_FILE} --content text

    echo "  生成 ${EXP_NAME} image code (GPU ${GPU})..."
    CUDA_VISIBLE_DEVICES=${GPU} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${IMG_IDX_FILE} --content image

    python -c "
import json
for name, path in [('text', '${DATA_DIR}/${IDX_FILE}'), ('image', '${DATA_DIR}/${IMG_IDX_FILE}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    print(f'  {name}: {len(codes)} items, {len(set(codes))} unique, collision={(1-len(set(codes))/len(codes))*100:.2f}%')
"
}

generate_indices "wo-concat_llama-caq" ${GPU_WO_CONCAT}
generate_indices "wo-caq_llama-concat" ${GPU_WO_CAQ}

echo ""
echo "=============================================="
echo " Games RQVAE + 索引全部完成! $(date)"
echo "=============================================="
