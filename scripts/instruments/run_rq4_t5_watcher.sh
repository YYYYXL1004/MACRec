#!/bin/bash
# RQ4 T5 自动监控启动脚本
# 管理 7 组 T5 任务:
#   (a) λ_r ∈ {0.5, 1.0, 5.0}   → 等 RQVAE 完成
#   (c) λ_a ∈ {0.0001, 0.01}    → 等 RQVAE 完成
#   (b) λ_p ∈ {0.002, 0.01}     → 立即可跑 (复用 E3v2 RQVAE)
#   注: λ_p=0 与 RQ2 "w/o CPA" 相同, 不重复跑
# 扫描全部 8 张 GPU, 空闲 >=14GB 的双卡启动 T5
# 用法: nohup bash scripts/instruments/run_rq4_t5_watcher.sh > run_rq4_t5_watcher.out 2>&1 &

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

E3V2_RQVAE="E3v2_llama-concat-caq"

FREE_MEM_THRESHOLD=14000  # MB
BASE_PORT=29730

# 任务定义: LABEL|RQVAE_TAG|T5_EXTRA_ARGS
# RQVAE_TAG 决定索引文件前缀: .index_lemb_${RQVAE_TAG}.json
declare -a TASKS_LIST
# (a) λ_r 敏感性 — 等 RQVAE
TASKS_LIST[0]="car0.5|rq4-car0.5|--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
TASKS_LIST[1]="car1.0|rq4-car1.0|--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
TASKS_LIST[2]="car5.0|rq4-car5.0|--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
# (c) λ_a 敏感性 — 等 RQVAE
TASKS_LIST[3]="align0.0001|rq4-align0.0001|--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
TASKS_LIST[4]="align0.01|rq4-align0.01|--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
# (b) λ_p 敏感性 — 立即可跑 (复用 E3v2 RQVAE)
TASKS_LIST[5]="cpa0.002|${E3V2_RQVAE}|--cpa_weight 0.002 --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
TASKS_LIST[6]="cpa0.01|${E3V2_RQVAE}|--cpa_weight 0.01 --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"

# 状态: 0=等RQVAE, 1=就绪等GPU, 2=进行中, 3=完成
declare -A STATUS
declare -A PIDS
declare -A GPU_USED_BY
PORT_OFFSET=0

for i in "${!TASKS_LIST[@]}"; do
    # 索引 0-4 等 RQVAE, 5-6 立即就绪
    if [ "$i" -le 4 ]; then
        STATUS[$i]=0
    else
        STATUS[$i]=1
    fi
    PIDS[$i]=0
done

cd $BASE_DIR

# 检查 RQVAE 索引是否存在
check_rqvae_done() {
    local rqvae_tag=$1
    local text_idx="${DATA_DIR}/${DATASET}.index_lemb_${rqvae_tag}.json"
    local image_idx="${DATA_DIR}/${DATASET}.index_vitemb_${rqvae_tag}.json"
    [ -f "$text_idx" ] && [ -f "$image_idx" ]
}

# 查找空闲双卡 (>= 阈值), 排除正在使用的
find_free_gpu_pair() {
    local used_gpus=""
    for i in "${!TASKS_LIST[@]}"; do
        if [ "${STATUS[$i]}" -eq 2 ] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
            used_gpus="${used_gpus} ${GPU_USED_BY[$i]:-}"
        fi
    done

    local free_gpus=()
    while IFS=',' read -r idx mem_free; do
        idx=$(echo "$idx" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
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

# T5 全流程: 训练 → 推理 → ensemble
run_t5_pipeline() {
    local label=$1
    local rqvae_tag=$2
    local gpus=$3
    local port=$4
    shift 4
    local extra_args="$@"

    local INDEX_FILE=".index_lemb_${rqvae_tag}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${rqvae_tag}.json"
    local OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-rq4-${label}-b${NUM_BEAMS}"

    echo ""
    echo "================================================================"
    echo " [rq4-${label}] T5: GPUs=${gpus}, port=${port}"
    echo " RQVAE: ${rqvae_tag} | 输出: ${OUTPUT_DIR}"
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
        $extra_args \
        2>&1 | tee $OUTPUT_DIR/train.log

    if [ $? -ne 0 ]; then
        echo " [rq4-${label}] T5 训练失败!"
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

    echo " [rq4-${label}] 完成! $(date)"
    echo " 结果: $(cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null)"
    return 0
}

echo "=============================================="
echo " RQ4 T5 自动监控"
echo " 扫描间隔: 60s | GPU: 0-7 | 显存阈值: ${FREE_MEM_THRESHOLD}MB"
echo " 共 ${#TASKS_LIST[@]} 组 T5 任务"
echo " 开始: $(date)"
echo "=============================================="

while true; do
    ALL_DONE=true

    for i in "${!TASKS_LIST[@]}"; do
        IFS='|' read -r label rqvae_tag extra_args <<< "${TASKS_LIST[$i]}"

        case ${STATUS[$i]} in
            0)
                ALL_DONE=false
                if check_rqvae_done "$rqvae_tag"; then
                    echo "[$(date '+%H:%M:%S')] [${label}] RQVAE 索引就绪"
                    STATUS[$i]=1
                fi
                ;;
            1)
                ALL_DONE=false
                GPU_PAIR=$(find_free_gpu_pair)
                if [ $? -eq 0 ]; then
                    PORT=$((BASE_PORT + PORT_OFFSET))
                    PORT_OFFSET=$((PORT_OFFSET + 1))
                    echo "[$(date '+%H:%M:%S')] [${label}] GPU: ${GPU_PAIR}, port=${PORT}"

                    GPU_USED_BY[$i]="${GPU_PAIR//,/ }"
                    run_t5_pipeline "$label" "$rqvae_tag" "$GPU_PAIR" "$PORT" $extra_args &
                    PIDS[$i]=$!
                    STATUS[$i]=2
                    sleep 30
                fi
                ;;
            2)
                ALL_DONE=false
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    wait "${PIDS[$i]}"
                    EXIT_CODE=$?
                    echo "[$(date '+%H:%M:%S')] [${label}] 退出 (exit=${EXIT_CODE})"
                    STATUS[$i]=3
                    unset GPU_USED_BY[$i]
                fi
                ;;
            3) ;;
        esac
    done

    # 打印状态
    echo -n "[$(date '+%H:%M:%S')] 状态: "
    for i in "${!TASKS_LIST[@]}"; do
        IFS='|' read -r label _ _ <<< "${TASKS_LIST[$i]}"
        case ${STATUS[$i]} in
            0) echo -n "${label}=等RQVAE " ;;
            1) echo -n "${label}=等GPU " ;;
            2) echo -n "${label}=T5中(${PIDS[$i]}) " ;;
            3) echo -n "${label}=完成 " ;;
        esac
    done
    echo ""

    if $ALL_DONE; then
        echo ""
        echo "=============================================="
        echo " RQ4 T5 全部完成! $(date)"
        echo "=============================================="
        break
    fi

    sleep 60
done
