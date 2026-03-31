#!/bin/bash
# RQ4(a)(c): 并行 RQVAE 超参训练 + 索引生成
# (a) CAR weight λ_r ∈ {0.5, 1.0, 5.0}   (baseline λ_r=2.0)
# (c) align weight λ_a ∈ {0.0001, 0.01}   (baseline λ_a=0.001)
# 共 5 组 RQVAE, 每组单卡 ~4GB, 自动寻找空闲 GPU 并行
# 用法: nohup bash scripts/instruments/run_rq4_rqvae_parallel.sh > run_rq4_rqvae_parallel.out 2>&1 &

eval "$(conda shell.bash hook)"
conda activate ETEGRec
export WANDB_MODE=disabled

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

FREE_MEM_THRESHOLD=8000  # RQVAE 单卡 ~4GB

# 5 组实验: TAG|λ_r|λ_a
EXPERIMENTS=(
    "car0.5|0.5|0.001"
    "car1.0|1.0|0.001"
    "car5.0|5.0|0.001"
    "align0.0001|2.0|0.0001"
    "align0.01|2.0|0.01"
)

cd ${BASE_DIR}/cross_index

# 自动寻找 N 张空闲 GPU (>阈值)
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
    echo "${gpus[@]}"
}

GPU_LIST=$(find_free_gpus ${#EXPERIMENTS[@]})
GPU_ARR=($GPU_LIST)
AVAILABLE=${#GPU_ARR[@]}

if [ "$AVAILABLE" -eq 0 ]; then
    echo "没有空闲 GPU (>=${FREE_MEM_THRESHOLD}MB), 退出"
    exit 1
fi

echo "=============================================="
echo " RQ4 RQVAE 并行训练: ${#EXPERIMENTS[@]} 组实验"
echo " 可用 GPU: ${GPU_LIST} (${AVAILABLE} 张)"
echo " 开始: $(date)"
echo "=============================================="

# 单组 RQVAE 训练 + 索引生成
run_rqvae_single() {
    local TAG=$1
    local CAR_W=$2
    local ALIGN_W=$3
    local GPU=$4

    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/rq4-${TAG}"
    local IDX_FILE="${DATASET}.index_lemb_rq4-${TAG}.json"
    local IMG_IDX_FILE="${DATASET}.index_vitemb_rq4-${TAG}.json"

    echo "[${TAG}] 开始: GPU=${GPU}, λ_r=${CAR_W}, λ_a=${ALIGN_W} $(date)"
    mkdir -p $RQVAE_DIR

    CUDA_VISIBLE_DEVICES=${GPU} python -u main.py \
        --device cuda:0 \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} --collab_fusion concat \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight ${CAR_W} \
        --ckpt_dir $RQVAE_DIR \
        --num_emb_list 256 256 256 256 --e_dim 32 \
        --sk_epsilons 0.0 0.0 0.0 0.0 --eval_step 2 --batch_size 2048 \
        --begin_cross_layer 2 --use_cross_rq True \
        --text_class_info ${TEXT_CLASS} --image_class_info ${IMAGE_CLASS} \
        --text_contrast_weight 0.1 --image_contrast_weight 0.1 \
        --recon_contrast_weight ${ALIGN_W} \
        --epochs 1000 2>&1 | tee $RQVAE_DIR/train.log

    if [ ! -f "$RQVAE_DIR/best_text_collision_model.pth" ]; then
        echo "[${TAG}] RQVAE 训练失败!"
        return 1
    fi

    # 索引生成
    CUDA_VISIBLE_DEVICES=${GPU} python -u generate_indices_distance.py \
        --dataset $DATASET --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${IDX_FILE} --content text

    CUDA_VISIBLE_DEVICES=${GPU} python -u generate_indices_distance.py \
        --dataset $DATASET --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${IMG_IDX_FILE} --content image

    # 碰撞率
    python -c "
import json
for name, path in [('text', '${DATA_DIR}/${IDX_FILE}'), ('image', '${DATA_DIR}/${IMG_IDX_FILE}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    print(f'  [${TAG}] {name}: {len(codes)} items, {len(set(codes))} unique, collision={(1-len(set(codes))/len(codes))*100:.2f}%')
"

    echo "[${TAG}] 完成! $(date)"
    return 0
}

# 并行启动, 分批 (GPU 数量可能少于实验数量)
declare -A PIDS
BATCH_SIZE=${AVAILABLE}
TOTAL=${#EXPERIMENTS[@]}
IDX=0

while [ $IDX -lt $TOTAL ]; do
    # 本批次的实验
    BATCH_END=$((IDX + BATCH_SIZE))
    [ $BATCH_END -gt $TOTAL ] && BATCH_END=$TOTAL

    echo ""
    echo "--- 批次: 实验 $((IDX+1))~${BATCH_END} / ${TOTAL} ---"

    for ((i=IDX; i<BATCH_END; i++)); do
        IFS='|' read -r tag car_w align_w <<< "${EXPERIMENTS[$i]}"
        gpu_idx=$(( (i - IDX) % AVAILABLE ))
        gpu=${GPU_ARR[$gpu_idx]}

        run_rqvae_single "$tag" "$car_w" "$align_w" "$gpu" &
        PIDS[$i]=$!
        echo "  启动 ${tag} → GPU ${gpu} (PID: ${PIDS[$i]})"
    done

    # 等待本批次完成
    for ((i=IDX; i<BATCH_END; i++)); do
        wait ${PIDS[$i]}
        IFS='|' read -r tag _ _ <<< "${EXPERIMENTS[$i]}"
        echo "  ${tag} 退出 (exit=$?)"
    done

    IDX=$BATCH_END
done

echo ""
echo "=============================================="
echo " RQ4 RQVAE 全部完成! $(date)"
echo "=============================================="
