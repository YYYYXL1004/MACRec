#!/bin/bash
# RQ3 T5 自动监控启动脚本
# 持续扫描: 哪个 fusion 的 RQVAE+索引已完成 且 有双卡 >=14GB 空闲 → 立即启动 T5
# 用法: nohup bash scripts/instruments/run_rq3_t5_watcher.sh > run_rq3_t5_watcher.out 2>&1 &

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50
CPA_W=0.005
CPA_LOSS=cosine

FUSION_TYPES=("proj_add" "gating" "cross_attn")
BASE_PORT=29700
FREE_MEM_THRESHOLD=14000  # MB

# 各 fusion 类型的状态: 0=等待RQVAE, 1=等待GPU, 2=T5进行中, 3=完成
declare -A STATUS
declare -A T5_PIDS
for ft in "${FUSION_TYPES[@]}"; do
    STATUS[$ft]=0
    T5_PIDS[$ft]=0
done

cd $BASE_DIR

# 检查 RQVAE+索引是否完成
check_rqvae_done() {
    local ft=$1
    local tag="rq3-${ft}"
    local text_idx="${DATA_DIR}/${DATASET}.index_lemb_${tag}.json"
    local image_idx="${DATA_DIR}/${DATASET}.index_vitemb_${tag}.json"
    # 索引文件存在 → RQVAE+索引全部完成
    [ -f "$text_idx" ] && [ -f "$image_idx" ]
}

# 查找两张空闲显存 >= 阈值的 GPU (排除已被本脚本 T5 占用的)
find_free_gpu_pair() {
    local used_gpus=""
    for ft in "${FUSION_TYPES[@]}"; do
        if [ "${STATUS[$ft]}" -eq 2 ] && kill -0 "${T5_PIDS[$ft]}" 2>/dev/null; then
            # 正在跑 T5, 通过 /proc 读 CUDA_VISIBLE_DEVICES 或直接标记
            used_gpus="${used_gpus} ${GPU_USED_BY[$ft]:-}"
        fi
    done

    local free_gpus=()
    while IFS=',' read -r idx mem_free; do
        idx=$(echo "$idx" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
        # 只考虑 GPU 0-3
        if [ "$idx" -gt 3 ]; then continue; fi
        # 跳过已被 T5 占用的 GPU
        if echo "$used_gpus" | grep -qw "$idx"; then continue; fi
        if [ "$mem_free" -ge "$FREE_MEM_THRESHOLD" ]; then
            free_gpus+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    if [ "${#free_gpus[@]}" -ge 2 ]; then
        echo "${free_gpus[0]},${free_gpus[1]}"
        return 0
    fi
    return 1
}

# T5 全流程: 训练 → 推理(seqrec + seqimage) → ensemble
run_t5_pipeline() {
    local ft=$1
    local gpus=$2
    local port=$3
    local tag="rq3-${ft}"
    local INDEX_FILE=".index_lemb_${tag}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${tag}.json"
    local OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-${tag}-b${NUM_BEAMS}"

    echo ""
    echo "================================================================"
    echo " [${tag}] T5 流程启动: GPUs=${gpus}, port=${port}"
    echo " 输出: ${OUTPUT_DIR}"
    echo " 开始: $(date)"
    echo "================================================================"

    mkdir -p $OUTPUT_DIR

    # T5 训练
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        finetune_contrastive.py \
        --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
        --base_model ./config/ckpt \
        --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
        --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
        --max_his_len 20 --prompt_num 4 --patient 10 \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --tasks $TASKS --valid_task seqrec \
        --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS \
        2>&1 | tee $OUTPUT_DIR/train.log

    if [ $? -ne 0 ]; then
        echo " [${tag}] T5 训练失败!"
        return 1
    fi

    # 推理 seqrec
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

    # 推理 seqimage
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

    # Ensemble
    python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET \
        --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $NUM_BEAMS

    echo ""
    echo " [${tag}] 全部完成! $(date)"
    echo " 结果: $(cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null)"
    return 0
}

echo "============================================"
echo " RQ3 T5 自动监控启动"
echo " 扫描间隔: 60s"
echo " 显存阈值: ${FREE_MEM_THRESHOLD}MB"
echo " 开始: $(date)"
echo "============================================"

declare -A GPU_USED_BY
PORT_OFFSET=0

while true; do
    ALL_DONE=true

    for ft in "${FUSION_TYPES[@]}"; do
        case ${STATUS[$ft]} in
            0)
                # 等待 RQVAE+索引完成
                ALL_DONE=false
                if check_rqvae_done "$ft"; then
                    echo "[$(date '+%H:%M:%S')] [rq3-${ft}] RQVAE+索引 已完成，等待 GPU..."
                    STATUS[$ft]=1
                fi
                ;;
            1)
                # RQVAE 完成，等待空闲 GPU
                ALL_DONE=false
                GPU_PAIR=$(find_free_gpu_pair)
                if [ $? -eq 0 ]; then
                    PORT=$((BASE_PORT + PORT_OFFSET))
                    PORT_OFFSET=$((PORT_OFFSET + 1))
                    echo "[$(date '+%H:%M:%S')] [rq3-${ft}] 找到空闲 GPU: ${GPU_PAIR}, 启动 T5 (port=${PORT})..."

                    # 记录占用的 GPU
                    GPU_USED_BY[$ft]="${GPU_PAIR//,/ }"

                    # 后台启动 T5 全流程
                    run_t5_pipeline "$ft" "$GPU_PAIR" "$PORT" &
                    T5_PIDS[$ft]=$!
                    STATUS[$ft]=2

                    # 等一下让 GPU 被占住，避免下一个也分配到同一对
                    sleep 30
                fi
                ;;
            2)
                # T5 进行中，检查是否完成
                ALL_DONE=false
                if ! kill -0 "${T5_PIDS[$ft]}" 2>/dev/null; then
                    wait "${T5_PIDS[$ft]}"
                    EXIT_CODE=$?
                    if [ $EXIT_CODE -eq 0 ]; then
                        echo "[$(date '+%H:%M:%S')] [rq3-${ft}] T5 流程完成! (exit=${EXIT_CODE})"
                    else
                        echo "[$(date '+%H:%M:%S')] [rq3-${ft}] T5 流程异常退出! (exit=${EXIT_CODE})"
                    fi
                    STATUS[$ft]=3
                    unset GPU_USED_BY[$ft]
                fi
                ;;
            3)
                # 已完成
                ;;
        esac
    done

    # 打印状态
    echo -n "[$(date '+%H:%M:%S')] 状态: "
    for ft in "${FUSION_TYPES[@]}"; do
        case ${STATUS[$ft]} in
            0) echo -n "${ft}=等RQVAE " ;;
            1) echo -n "${ft}=等GPU " ;;
            2) echo -n "${ft}=T5中(PID:${T5_PIDS[$ft]}) " ;;
            3) echo -n "${ft}=完成 " ;;
        esac
    done
    echo ""

    # 全部完成则退出
    if $ALL_DONE; then
        echo ""
        echo "============================================"
        echo " RQ3 T5 全部完成! $(date)"
        echo "============================================"
        break
    fi

    sleep 60
done
