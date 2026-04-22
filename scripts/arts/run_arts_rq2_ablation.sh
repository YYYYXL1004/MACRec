#!/bin/bash
# ============================================================================
# Arts RQ2 消融实验: 自动 GPU 调度 + 串并行优化
# ============================================================================
# 4 组消融 (对照: R2_llama-concat-caq + LCPA 0.005 + Aug + beam-50):
#   消融1 (w/o Input Anchoring): RQVAE 去掉 collab concat, 保留 CAR
#   消融2 (w/o CAR):             RQVAE 保留 concat, 去掉 CAQ (weight=0)
#   消融3 (w/o CPA):             复用 R2 RQVAE, T5 不加 LCPA
#   消融4 (w/o Reranking):       复用完整 T5, beam-20 推理
#
# GPU 调度策略:
#   RQVAE 训练: 任意 >8GB 空闲卡 (单卡 ~4GB)
#   T5 训练/推理: 需要一对 >14GB 空闲卡 (双卡 DDP)
#
# 执行计划 (最大化并行):
#   Phase A: 并行训练 2 个 RQVAE (消融1+2, 各占 1 张小卡)
#          + 同时 T5 训练消融3 w/o CPA (2 张大卡)
#   Phase B: 消融4 beam-20 推理 (等消融3的GPU释放, 或复用)
#   Phase C: RQVAE 完成后生成索引
#   Phase D: 并行/串行训练消融1+2 的 T5
#
# 用法: nohup bash scripts/arts/run_arts_rq2_ablation.sh > run_arts_rq2_ablation.out 2>&1 &
# ============================================================================

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# ======================== 全局配置 ========================
DATASET="Arts"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50
CPA_W=0.005
CPA_LOSS=cosine
AUG_DROPOUT=0.1
AUG_CROP=0.3

# 完整模型
FULL_RQVAE="R2_llama-concat-caq"
FULL_T5_DIR="${BASE_DIR}/log/${DATASET}-${FULL_RQVAE}-lcpa${CPA_W}-aug-b20"

# 消融 RQVAE 名
WO_CONCAT_RQVAE="wo-concat_llama-caq"
WO_CAQ_RQVAE="wo-caq_llama-concat"

# GPU 阈值
RQVAE_MEM_THRESHOLD=8000   # RQVAE 训练需要 >8GB
T5_MEM_THRESHOLD=14000      # T5 训练/推理需要 >14GB

# 端口计数器
PORT_COUNTER=29700

cd $BASE_DIR

# ======================== GPU 调度函数 ========================
# 查找 N 张空闲 GPU (大于指定显存阈值), 排除指定 GPU
find_free_gpus() {
    local needed=$1
    local threshold=$2
    shift 2
    local exclude_str="$*"
    
    local gpus=()
    while IFS=',' read -r idx mem_free; do
        idx=$(echo "$idx" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
        # 排除正在使用的 GPU
        local skip=false
        for ex in $exclude_str; do
            [ "$idx" = "$ex" ] && skip=true && break
        done
        $skip && continue
        if [ "$mem_free" -ge "$threshold" ]; then
            gpus+=("$idx")
        fi
        [ "${#gpus[@]}" -ge "$needed" ] && break
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    
    if [ "${#gpus[@]}" -lt "$needed" ]; then
        return 1
    fi
    echo "${gpus[@]}"
}

next_port() {
    PORT_COUNTER=$((PORT_COUNTER + 1))
    echo $PORT_COUNTER
}

# ======================== 通用流程函数 ========================
# RQVAE 训练
train_rqvae() {
    local label=$1
    local gpu=$2
    local ckpt_dir=$3
    local collab_data_arg=$4       # collab_data_path 或空
    local collab_fusion_arg=$5     # concat 或空
    local collab_cw=$6             # contrastive weight
    local collab_neighbor_arg=$7   # neighbor info 或空
    
    mkdir -p $ckpt_dir
    
    echo ""
    echo "================================================================"
    echo " [${label}] RQVAE 训练 | GPU ${gpu}"
    echo " collab_data=${collab_data_arg:-无}"
    echo " collab_fusion=${collab_fusion_arg:-无}"
    echo " collab_contrastive_weight=${collab_cw}"
    echo " 开始: $(date)"
    echo "================================================================"
    
    local extra_args=""
    [ -n "$collab_data_arg" ] && extra_args="${extra_args} --collab_data_path ${collab_data_arg} --collab_fusion ${collab_fusion_arg}"
    [ -n "$collab_neighbor_arg" ] && extra_args="${extra_args} --collab_neighbor_info ${collab_neighbor_arg}"
    
    cd ${BASE_DIR}/cross_index
    
    CUDA_VISIBLE_DEVICES=${gpu} python -u main.py \
        --device cuda:0 \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --collab_contrastive_weight ${collab_cw} \
        --ckpt_dir ${ckpt_dir} \
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
        --epochs 1000 \
        ${extra_args} 2>&1 | tee ${ckpt_dir}/train.log
    
    cd $BASE_DIR
    
    if [ ! -f "${ckpt_dir}/best_text_collision_model.pth" ] || [ ! -f "${ckpt_dir}/best_image_collision_model.pth" ]; then
        echo "[${label}] RQVAE 训练失败！模型文件缺失"
        return 1
    fi
    echo "[${label}] RQVAE 训练完成 $(date)"
    return 0
}

# 生成离散索引
generate_indices() {
    local label=$1
    local rqvae_name=$2
    local gpu=$3
    local rqvae_dir="${BASE_DIR}/cross_index/log/${DATASET}/${rqvae_name}"
    local idx_file="${DATASET}.index_lemb_${rqvae_name}.json"
    local img_idx_file="${DATASET}.index_vitemb_${rqvae_name}.json"
    
    cd ${BASE_DIR}/cross_index
    
    echo "[${label}] 生成 text code (GPU ${gpu})... $(date)"
    CUDA_VISIBLE_DEVICES=${gpu} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${rqvae_dir}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${idx_file} --content text
    
    echo "[${label}] 生成 image code (GPU ${gpu})... $(date)"
    CUDA_VISIBLE_DEVICES=${gpu} python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${rqvae_dir}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${img_idx_file} --content image
    
    cd $BASE_DIR
    
    python -c "
import json
for name, path in [('text', '${DATA_DIR}/${idx_file}'), ('image', '${DATA_DIR}/${img_idx_file}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    print(f'  [${label}] {name}: {len(codes)} items, {len(set(codes))} unique, collision={(1-len(set(codes))/len(codes))*100:.2f}%')
"
}

# T5 全流程: 训练 → 推理 → Ensemble
run_t5_pipeline() {
    local label=$1
    local rqvae_name=$2
    local output_dir=$3
    local gpus=$4
    local port=$5
    shift 5
    local extra_args="$@"
    
    local INDEX_FILE=".index_lemb_${rqvae_name}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${rqvae_name}.json"
    
    echo ""
    echo "================================================================"
    echo " [${label}] T5 全流程 | GPUs=${gpus}, port=${port}"
    echo " RQVAE: ${rqvae_name}"
    echo " 输出: ${output_dir}"
    echo " 额外: ${extra_args}"
    echo " 开始: $(date)"
    echo "================================================================"
    
    mkdir -p $output_dir
    
    # T5 训练
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        finetune_contrastive.py \
        --data_path ./data/ --dataset $DATASET --output_dir $output_dir \
        --base_model ./config/ckpt \
        --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
        --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
        --max_his_len 20 --prompt_num 4 --patient 10 \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --tasks $TASKS --valid_task seqrec \
        --aug_item_dropout $AUG_DROPOUT --aug_crop_prob $AUG_CROP \
        $extra_args 2>&1 | tee $output_dir/train.log
    
    if [ ! -f "$output_dir/model.safetensors" ] && [ ! -f "$output_dir/pytorch_model.bin" ]; then
        echo "[${label}] T5 训练失败！"
        return 1
    fi
    echo "[${label}] T5 训练完成 $(date)"
    
    # 推理 + Ensemble
    run_inference "$label" "$output_dir" "$INDEX_FILE" "$IMAGE_INDEX_FILE" "$gpus" "$port" $NUM_BEAMS
}

# 仅推理 + Ensemble
run_inference() {
    local label=$1
    local ckpt_dir=$2
    local index_file=$3
    local image_index_file=$4
    local gpus=$5
    local port=$6
    local beams=$7
    
    echo "[${label}] seqrec 推理 (beam-${beams})... $(date)"
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $ckpt_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $beams \
        --index_file $index_file --image_index_file $image_index_file \
        --test_task seqrec \
        --results_file $ckpt_dir/results_seqrec_${beams}.json \
        --save_file $ckpt_dir/save_seqrec_${beams}.json --filter_items
    
    echo "[${label}] seqimage 推理 (beam-${beams})... $(date)"
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $ckpt_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $beams \
        --index_file $index_file --image_index_file $image_index_file \
        --test_task seqimage \
        --results_file $ckpt_dir/results_seqimage_${beams}.json \
        --save_file $ckpt_dir/save_seqimage_${beams}.json --filter_items
    
    echo "[${label}] Ensemble (beam-${beams})... $(date)"
    python ensemble.py --output_dir $ckpt_dir --dataset $DATASET \
        --data_path ./data/ --index_file $index_file --image_index_file $image_index_file \
        --num_beams $beams
    
    echo ""
    echo "=== [${label}] 结果 ==="
    cat $ckpt_dir/results_ensemble_${beams}.json 2>/dev/null
    echo ""
}

# 等待一对空闲 T5 GPU (轮询)
# 注意: 不能写 local pair=$(cmd), bash 的 local 会覆盖 $? 为 0
wait_for_t5_gpus() {
    local exclude="$1"
    local pair
    while true; do
        pair=$(find_free_gpus 2 $T5_MEM_THRESHOLD $exclude)
        if [ $? -eq 0 ]; then
            echo "$pair"
            return 0
        fi
        echo "[wait_for_t5_gpus] 未找到足够空闲GPU (>$T5_MEM_THRESHOLD MiB), 30秒后重试..." >&2
        sleep 30
    done
}

# ============================================================================
echo "============================================================"
echo " Arts RQ2 消融实验 (4组)"
echo " 完整模型 RQVAE: ${FULL_RQVAE}"
echo " 完整模型 T5: ${FULL_T5_DIR}"
echo " 开始: $(date)"
echo "============================================================"

# ================================================================
# Phase A: 并行启动 RQVAE (消融1+2) + 同时启动 w/o CPA T5 (消融3)
# ================================================================
echo ""
echo "================================================================"
echo " Phase A: 并行启动 RQVAE×2 + T5 w/o CPA"
echo " $(date)"
echo "================================================================"

# 追踪已占用的 GPU
USED_GPUS=""

# --- 启动 RQVAE 消融1: w/o Input Anchoring (无 collab concat, 保留 CAR) ---
RQVAE_GPUS=$(find_free_gpus 2 $RQVAE_MEM_THRESHOLD $USED_GPUS)
RQVAE_GPU_ARR=($RQVAE_GPUS)
GPU_RQVAE1=${RQVAE_GPU_ARR[0]}
GPU_RQVAE2=${RQVAE_GPU_ARR[1]}
USED_GPUS="${USED_GPUS} ${GPU_RQVAE1} ${GPU_RQVAE2}"

WO_CONCAT_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CONCAT_RQVAE}"
train_rqvae "w/o InputAnchoring RQVAE" "$GPU_RQVAE1" "$WO_CONCAT_RQVAE_DIR" \
    "" "" "2.0" "${COLLAB_NEIGHBOR}" &
PID_RQVAE1=$!

# --- 启动 RQVAE 消融2: w/o CAR (保留 concat, 去掉 CAQ) ---
WO_CAQ_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CAQ_RQVAE}"
train_rqvae "w/o CAR RQVAE" "$GPU_RQVAE2" "$WO_CAQ_RQVAE_DIR" \
    "${COLLAB}" "concat" "0.0" "" &
PID_RQVAE2=$!

echo "[Phase A] RQVAE 已启动: w/o concat GPU=${GPU_RQVAE1} (PID=${PID_RQVAE1}), w/o CAQ GPU=${GPU_RQVAE2} (PID=${PID_RQVAE2})"

# --- 同时启动消融3: w/o CPA T5 训练 ---
# RQVAE 只用 ~1.3G, 不需要排除 RQVAE 所在的 GPU, 靠 memory threshold 自然过滤
sleep 5  # 等 RQVAE 分配显存后再查
T5_GPUS_STR=$(wait_for_t5_gpus "")
T5_GPU_ARR=($T5_GPUS_STR)
T5_GPU_PAIR="${T5_GPU_ARR[0]},${T5_GPU_ARR[1]}"

WO_CPA_DIR="${BASE_DIR}/log/${DATASET}-wo-cpa-aug-b${NUM_BEAMS}"
PORT_CPA=$(next_port)

echo "[Phase A] w/o CPA T5 启动: GPUs=${T5_GPU_PAIR}, port=${PORT_CPA}"
run_t5_pipeline "w/o CPA" "$FULL_RQVAE" "$WO_CPA_DIR" "$T5_GPU_PAIR" "$PORT_CPA" &
PID_T5_CPA=$!

echo ""
echo "[Phase A] 状态汇总:"
echo "  RQVAE w/o concat: GPU ${GPU_RQVAE1}, PID ${PID_RQVAE1}"
echo "  RQVAE w/o CAQ:    GPU ${GPU_RQVAE2}, PID ${PID_RQVAE2}"
echo "  T5 w/o CPA:       GPU ${T5_GPU_PAIR}, PID ${PID_T5_CPA}"

# ================================================================
# Phase B: 等 w/o CPA 完成后, 立刻跑消融4 beam-20
# ================================================================
echo ""
echo "================================================================"
echo " Phase B: 等待 w/o CPA 完成 → 启动消融4 beam-20"
echo "================================================================"

wait $PID_T5_CPA
echo "[Phase B] w/o CPA T5 已完成 $(date)"

# w/o CPA 释放了 GPU, 重新查找空闲卡跑 beam-20
FULL_INDEX=".index_lemb_${FULL_RQVAE}.json"
FULL_IMAGE_INDEX=".index_vitemb_${FULL_RQVAE}.json"
PORT_BEAM20=$(next_port)

BEAM20_GPUS_STR=$(wait_for_t5_gpus "")
BEAM20_GPU_ARR=($BEAM20_GPUS_STR)
BEAM20_GPU_PAIR="${BEAM20_GPU_ARR[0]},${BEAM20_GPU_ARR[1]}"

echo "[Phase B] 消融4: beam-20 推理启动 GPUs=${BEAM20_GPU_PAIR}, port=${PORT_BEAM20}"
run_inference "w/o Reranking (beam-20)" "$FULL_T5_DIR" \
    "$FULL_INDEX" "$FULL_IMAGE_INDEX" "$BEAM20_GPU_PAIR" "$PORT_BEAM20" 20
echo "[Phase B] 消融4 完成 $(date)"

# ================================================================
# Phase C: 等待 RQVAE 完成, 生成索引
# ================================================================
echo ""
echo "================================================================"
echo " Phase C: 等待 RQVAE 完成 → 生成索引"
echo "================================================================"

# RQVAE 可能已经完成了 (Phase A/B 耗时可能较长)
wait $PID_RQVAE1
EXIT_RQVAE1=$?
echo "[Phase C] RQVAE w/o concat 完成 (exit=${EXIT_RQVAE1}) $(date)"

wait $PID_RQVAE2
EXIT_RQVAE2=$?
echo "[Phase C] RQVAE w/o CAQ 完成 (exit=${EXIT_RQVAE2}) $(date)"

if [ $EXIT_RQVAE1 -ne 0 ] || [ $EXIT_RQVAE2 -ne 0 ]; then
    echo "[Phase C] 有 RQVAE 训练失败，检查日志!"
    # 不退出, 继续能跑的
fi

# 用 RQVAE 原来的卡生成索引 (串行, 很快)
generate_indices "w/o InputAnchoring" "$WO_CONCAT_RQVAE" "$GPU_RQVAE1"
generate_indices "w/o CAR" "$WO_CAQ_RQVAE" "$GPU_RQVAE1"
echo "[Phase C] 索引生成完成 $(date)"

# ================================================================
# Phase D: 并行或串行训练消融1+2 的 T5
# ================================================================
echo ""
echo "================================================================"
echo " Phase D: 消融1 (w/o InputAnchoring) + 消融2 (w/o CAR) T5 训练"
echo "================================================================"

# 尝试找 2 对 GPU 并行跑; 不够就串行
PAIR1=$(find_free_gpus 2 $T5_MEM_THRESHOLD)
if [ $? -eq 0 ]; then
    P1_ARR=($PAIR1)
    GPUS_ABL1="${P1_ARR[0]},${P1_ARR[1]}"
    EXCLUDE_P1="${P1_ARR[0]} ${P1_ARR[1]}"
    
    # 尝试找第二对
    sleep 5
    PAIR2=$(find_free_gpus 2 $T5_MEM_THRESHOLD $EXCLUDE_P1)
    if [ $? -eq 0 ]; then
        # 有 2 对空闲卡: 并行
        P2_ARR=($PAIR2)
        GPUS_ABL2="${P2_ARR[0]},${P2_ARR[1]}"
        
        PORT_ABL1=$(next_port)
        PORT_ABL2=$(next_port)
        
        echo "[Phase D] 并行模式: w/o InputAnchoring GPU=${GPUS_ABL1}, w/o CAR GPU=${GPUS_ABL2}"
        
        WO_CONCAT_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-aug-b${NUM_BEAMS}"
        run_t5_pipeline "w/o InputAnchoring" "$WO_CONCAT_RQVAE" "$WO_CONCAT_T5_DIR" \
            "$GPUS_ABL1" "$PORT_ABL1" \
            --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS &
        PID_ABL1=$!
        
        sleep 30
        
        WO_CAQ_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-aug-b${NUM_BEAMS}"
        run_t5_pipeline "w/o CAR" "$WO_CAQ_RQVAE" "$WO_CAQ_T5_DIR" \
            "$GPUS_ABL2" "$PORT_ABL2" \
            --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS &
        PID_ABL2=$!
        
        echo "[Phase D] PID: w/o InputAnchoring=${PID_ABL1}, w/o CAR=${PID_ABL2}"
        wait $PID_ABL1
        echo "[Phase D] w/o InputAnchoring 完成 $(date)"
        wait $PID_ABL2
        echo "[Phase D] w/o CAR 完成 $(date)"
    else
        # 只有 1 对空闲卡: 串行
        echo "[Phase D] 串行模式 (仅找到1对空闲GPU): GPUs=${GPUS_ABL1}"
        
        PORT_ABL1=$(next_port)
        WO_CONCAT_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-aug-b${NUM_BEAMS}"
        run_t5_pipeline "w/o InputAnchoring" "$WO_CONCAT_RQVAE" "$WO_CONCAT_T5_DIR" \
            "$GPUS_ABL1" "$PORT_ABL1" \
            --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS
        
        PORT_ABL2=$(next_port)
        WO_CAQ_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-aug-b${NUM_BEAMS}"
        run_t5_pipeline "w/o CAR" "$WO_CAQ_RQVAE" "$WO_CAQ_T5_DIR" \
            "$GPUS_ABL1" "$PORT_ABL2" \
            --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS
    fi
else
    # 没有空闲卡, 轮询等待
    echo "[Phase D] 无空闲GPU, 轮询等待..."
    
    GPUS_ABL1_STR=$(wait_for_t5_gpus "")
    ABL1_ARR=($GPUS_ABL1_STR)
    GPUS_ABL1="${ABL1_ARR[0]},${ABL1_ARR[1]}"
    
    PORT_ABL1=$(next_port)
    WO_CONCAT_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-aug-b${NUM_BEAMS}"
    run_t5_pipeline "w/o InputAnchoring" "$WO_CONCAT_RQVAE" "$WO_CONCAT_T5_DIR" \
        "$GPUS_ABL1" "$PORT_ABL1" \
        --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS
    
    PORT_ABL2=$(next_port)
    WO_CAQ_T5_DIR="${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-aug-b${NUM_BEAMS}"
    run_t5_pipeline "w/o CAR" "$WO_CAQ_RQVAE" "$WO_CAQ_T5_DIR" \
        "$GPUS_ABL1" "$PORT_ABL2" \
        --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS
fi

# ================================================================
# 汇总
# ================================================================
echo ""
echo "============================================================"
echo " Arts RQ2 消融实验全部完成! $(date)"
echo "============================================================"
echo ""
echo "=== 完整模型 (参考) ==="
cat "${FULL_T5_DIR}/results_ensemble_${NUM_BEAMS}.json" 2>/dev/null
echo ""

echo "=== 消融1: w/o Input Anchoring ==="
cat "${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-aug-b${NUM_BEAMS}/results_ensemble_${NUM_BEAMS}.json" 2>/dev/null
echo ""

echo "=== 消融2: w/o CAR ==="
cat "${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-aug-b${NUM_BEAMS}/results_ensemble_${NUM_BEAMS}.json" 2>/dev/null
echo ""

echo "=== 消融3: w/o CPA ==="
cat "${WO_CPA_DIR}/results_ensemble_${NUM_BEAMS}.json" 2>/dev/null
echo ""

echo "=== 消融4: w/o Reranking (beam-20) ==="
cat "${FULL_T5_DIR}/results_ensemble_20.json" 2>/dev/null
echo ""

echo "============================================================"
echo " 全部完成! 请查看上方各组 ensemble 结果"
echo " $(date)"
echo "============================================================"
