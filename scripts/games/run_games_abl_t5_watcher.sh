#!/bin/bash
# Games RQ2 消融: T5 自动监控启动脚本
# 管理 4 组消融的 T5 流程:
#   消融1 (w/o concat):  等 RQVAE → T5训练+推理+ensemble
#   消融2 (w/o CAQ):     等 RQVAE → T5训练+推理+ensemble
#   消融3 (w/o CPA):     立即可跑 → T5训练(无LCPA)+推理+ensemble
#   消融4 (w/o beam-50): 立即可跑 → 仅 beam-20 推理+ensemble
# 扫描全部 8 张 GPU, 空闲 >=14GB 的双卡启动 T5
# 用法: nohup bash scripts/games/run_games_abl_t5_watcher.sh > run_games_abl_t5_watcher.out 2>&1 &

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Games"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50
CPA_W=0.005
CPA_LOSS=cosine
AUG_DROPOUT=0.1
AUG_CROP=0.3

# Full 模型路径 (消融4 beam-20 用)
FULL_RQVAE="R2_llama-concat-caq"
FULL_T5_DIR="${BASE_DIR}/log/${DATASET}-${FULL_RQVAE}-lcpa${CPA_W}-aug-b${NUM_BEAMS}"

FREE_MEM_THRESHOLD=14000  # MB
BASE_PORT=29720

# 消融任务定义
# 格式: LABEL|RQVAE_NAME|OUTPUT_DIR|TYPE
# TYPE: train=T5全流程(训练+推理+ensemble), infer=仅推理+ensemble
declare -a ABL_TASKS
ABL_TASKS[0]="wo-concat|wo-concat_llama-caq|${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-aug-b${NUM_BEAMS}|train"
ABL_TASKS[1]="wo-caq|wo-caq_llama-concat|${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-aug-b${NUM_BEAMS}|train"
ABL_TASKS[2]="wo-cpa|${FULL_RQVAE}|${BASE_DIR}/log/${DATASET}-wo-cpa-aug-b${NUM_BEAMS}|train"
ABL_TASKS[3]="wo-beam50|${FULL_RQVAE}|${FULL_T5_DIR}|infer"

# 各任务状态: 0=等待RQVAE, 1=就绪等GPU, 2=进行中, 3=完成
declare -A STATUS
declare -A PIDS
declare -A GPU_USED_BY
PORT_OFFSET=0

for i in "${!ABL_TASKS[@]}"; do
    IFS='|' read -r label rqvae_name output_dir task_type <<< "${ABL_TASKS[$i]}"
    # 消融1,2 需等 RQVAE, 消融3,4 立即就绪
    if [ "$i" -le 1 ]; then
        STATUS[$i]=0
    else
        STATUS[$i]=1
    fi
    PIDS[$i]=0
done

cd $BASE_DIR

# 检查 RQVAE 索引是否存在
check_rqvae_done() {
    local rqvae_name=$1
    local text_idx="${DATA_DIR}/${DATASET}.index_lemb_${rqvae_name}.json"
    local image_idx="${DATA_DIR}/${DATASET}.index_vitemb_${rqvae_name}.json"
    [ -f "$text_idx" ] && [ -f "$image_idx" ]
}

# 查找一对空闲 GPU (>= 阈值), 排除正在使用的
find_free_gpu_pair() {
    local used_gpus=""
    for i in "${!ABL_TASKS[@]}"; do
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

# T5 全流程: 训练 → 推理(seqrec+seqimage) → ensemble
run_t5_train_pipeline() {
    local label=$1
    local rqvae_name=$2
    local output_dir=$3
    local gpus=$4
    local port=$5
    local extra_t5_args=$6

    local INDEX_FILE=".index_lemb_${rqvae_name}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${rqvae_name}.json"

    echo ""
    echo "================================================================"
    echo " [${label}] T5 训练流程启动: GPUs=${gpus}, port=${port}"
    echo " RQVAE索引: ${rqvae_name}"
    echo " 输出: ${output_dir}"
    echo " 额外参数: ${extra_t5_args}"
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
        $extra_t5_args \
        2>&1 | tee $output_dir/train.log

    if [ $? -ne 0 ]; then
        echo " [${label}] T5 训练失败!"
        return 1
    fi

    # 推理 seqrec
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $output_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $output_dir/results_seqrec_${NUM_BEAMS}.json \
        --save_file $output_dir/save_seqrec_${NUM_BEAMS}.json --filter_items

    # 推理 seqimage
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $output_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $output_dir/results_seqimage_${NUM_BEAMS}.json \
        --save_file $output_dir/save_seqimage_${NUM_BEAMS}.json --filter_items

    # Ensemble
    python ensemble.py --output_dir $output_dir --dataset $DATASET \
        --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $NUM_BEAMS

    echo " [${label}] 全部完成! $(date)"
    echo " 结果: $(cat $output_dir/results_ensemble_${NUM_BEAMS}.json 2>/dev/null)"
    return 0
}

# 仅推理: beam-20 推理 + ensemble (复用已有 T5 模型)
run_beam20_infer() {
    local label=$1
    local rqvae_name=$2
    local ckpt_dir=$3
    local gpus=$4
    local port=$5

    local INDEX_FILE=".index_lemb_${rqvae_name}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${rqvae_name}.json"
    local BEAM=20

    echo ""
    echo "================================================================"
    echo " [${label}] beam-20 推理: GPUs=${gpus}, port=${port}"
    echo " 模型: ${ckpt_dir}"
    echo " 开始: $(date)"
    echo "================================================================"

    # seqrec beam-20
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $ckpt_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $BEAM \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $ckpt_dir/results_seqrec_${BEAM}.json \
        --save_file $ckpt_dir/save_seqrec_${BEAM}.json --filter_items

    # seqimage beam-20
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=2 --master_port=$port \
        test_ddp_save.py \
        --ckpt_path $ckpt_dir --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $BEAM \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $ckpt_dir/results_seqimage_${BEAM}.json \
        --save_file $ckpt_dir/save_seqimage_${BEAM}.json --filter_items

    # Ensemble beam-20
    python ensemble.py --output_dir $ckpt_dir --dataset $DATASET \
        --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $BEAM

    echo " [${label}] beam-20 推理完成! $(date)"
    echo " 结果: $(cat $ckpt_dir/results_ensemble_${BEAM}.json 2>/dev/null)"
    return 0
}

echo "=============================================="
echo " Games RQ2 消融 T5 自动监控"
echo " 扫描间隔: 60s | GPU: 0-7 | 显存阈值: ${FREE_MEM_THRESHOLD}MB"
echo " 开始: $(date)"
echo "=============================================="

while true; do
    ALL_DONE=true

    for i in "${!ABL_TASKS[@]}"; do
        IFS='|' read -r label rqvae_name output_dir task_type <<< "${ABL_TASKS[$i]}"

        case ${STATUS[$i]} in
            0)
                # 等待 RQVAE 索引
                ALL_DONE=false
                if check_rqvae_done "$rqvae_name"; then
                    echo "[$(date '+%H:%M:%S')] [${label}] RQVAE 索引就绪，等待 GPU..."
                    STATUS[$i]=1
                fi
                ;;
            1)
                # 就绪，等空闲 GPU
                ALL_DONE=false
                GPU_PAIR=$(find_free_gpu_pair)
                if [ $? -eq 0 ]; then
                    PORT=$((BASE_PORT + PORT_OFFSET))
                    PORT_OFFSET=$((PORT_OFFSET + 1))
                    echo "[$(date '+%H:%M:%S')] [${label}] 找到 GPU: ${GPU_PAIR}, port=${PORT}"

                    GPU_USED_BY[$i]="${GPU_PAIR//,/ }"

                    if [ "$task_type" = "train" ]; then
                        # 确定额外参数
                        local_extra=""
                        if [ "$label" = "wo-cpa" ]; then
                            # w/o CPA: 不传 cpa_weight
                            local_extra=""
                        else
                            # 消融1,2: 保留 LCPA
                            local_extra="--cpa_weight ${CPA_W} --collab_emb_path ${COLLAB} --cpa_loss_type ${CPA_LOSS}"
                        fi
                        run_t5_train_pipeline "$label" "$rqvae_name" "$output_dir" "$GPU_PAIR" "$PORT" "$local_extra" &
                    else
                        # 消融4: 仅推理 beam-20
                        run_beam20_infer "$label" "$rqvae_name" "$output_dir" "$GPU_PAIR" "$PORT" &
                    fi
                    PIDS[$i]=$!
                    STATUS[$i]=2
                    # 等 30s 让 GPU 被占住
                    sleep 30
                fi
                ;;
            2)
                # 进行中
                ALL_DONE=false
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    wait "${PIDS[$i]}"
                    EXIT_CODE=$?
                    if [ $EXIT_CODE -eq 0 ]; then
                        echo "[$(date '+%H:%M:%S')] [${label}] 完成! (exit=${EXIT_CODE})"
                    else
                        echo "[$(date '+%H:%M:%S')] [${label}] 异常退出! (exit=${EXIT_CODE})"
                    fi
                    STATUS[$i]=3
                    unset GPU_USED_BY[$i]
                fi
                ;;
            3)
                ;;
        esac
    done

    # 打印状态
    echo -n "[$(date '+%H:%M:%S')] 状态: "
    for i in "${!ABL_TASKS[@]}"; do
        IFS='|' read -r label _ _ _ <<< "${ABL_TASKS[$i]}"
        case ${STATUS[$i]} in
            0) echo -n "${label}=等RQVAE " ;;
            1) echo -n "${label}=等GPU " ;;
            2) echo -n "${label}=进行中(PID:${PIDS[$i]}) " ;;
            3) echo -n "${label}=完成 " ;;
        esac
    done
    echo ""

    if $ALL_DONE; then
        echo ""
        echo "=============================================="
        echo " Games RQ2 消融 T5 全部完成! $(date)"
        echo "=============================================="
        break
    fi

    sleep 60
done
