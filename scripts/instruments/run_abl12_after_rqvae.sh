#!/bin/bash
# 消融1 (w/o concat) + 消融2 (w/o CAQ)
# 等待 RQVAE 训练完成 → 生成索引 → 并行启动 T5
# 消融1: GPU 0,1 | 消融2: GPU 2,3

eval "$(conda shell.bash hook)"
conda activate ETEGRec
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"
TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50

# RQVAE 进程 PID (由当前正在运行的进程)
RQVAE_PID_WO_CONCAT=2248754
RQVAE_PID_WO_CAQ=2248758

WO_CONCAT_RQVAE="wo-concat_llama-caq"
WO_CAQ_RQVAE="wo-caq_llama-concat"
WO_CONCAT_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CONCAT_RQVAE}"
WO_CAQ_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CAQ_RQVAE}"

cd $BASE_DIR

echo "============================================"
echo " 等待 RQVAE 训练完成..."
echo " wo-concat PID: ${RQVAE_PID_WO_CONCAT}"
echo " wo-caq    PID: ${RQVAE_PID_WO_CAQ}"
echo " $(date)"
echo "============================================"

# 等待两个 RQVAE 进程完成
while kill -0 $RQVAE_PID_WO_CONCAT 2>/dev/null || kill -0 $RQVAE_PID_WO_CAQ 2>/dev/null; do
    sleep 30
    # 打印存活状态
    alive=""
    kill -0 $RQVAE_PID_WO_CONCAT 2>/dev/null && alive="${alive} wo-concat(${RQVAE_PID_WO_CONCAT})"
    kill -0 $RQVAE_PID_WO_CAQ 2>/dev/null && alive="${alive} wo-caq(${RQVAE_PID_WO_CAQ})"
    echo "  [$(date +%H:%M:%S)] 仍在运行:${alive}"
done

echo "两个 RQVAE 训练均已完成 $(date)"

# 验证模型文件
for rdir in $WO_CONCAT_RQVAE_DIR $WO_CAQ_RQVAE_DIR; do
    if [ ! -f "$rdir/best_text_collision_model.pth" ] || [ ! -f "$rdir/best_image_collision_model.pth" ]; then
        echo "RQVAE 模型缺失: $rdir"
        exit 1
    fi
done

# ================================================================
# 生成索引
# ================================================================
echo ""
echo "================================================================"
echo " 生成离散索引"
echo "================================================================"

generate_indices() {
    local EXP_NAME=$1
    local GPU=$2
    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_NAME}"
    local IDX_FILE="${DATASET}.index_lemb_${EXP_NAME}.json"
    local IMG_IDX_FILE="${DATASET}.index_vitemb_${EXP_NAME}.json"

    cd ${BASE_DIR}/cross_index

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

    cd $BASE_DIR

    python -c "
import json
for name, path in [('text', '${DATA_DIR}/${IDX_FILE}'), ('image', '${DATA_DIR}/${IMG_IDX_FILE}')]:
    d = json.load(open(path))
    codes = [tuple(v) for v in d.values()]
    unique = len(set(codes))
    total = len(codes)
    collision = (1 - unique/total) * 100
    print(f'  {name}: {total} items, {unique} unique codes, collision={collision:.2f}%')
"
}

generate_indices $WO_CONCAT_RQVAE 0
generate_indices $WO_CAQ_RQVAE 0

echo "索引生成完成 $(date)"

# ================================================================
# 通用 T5 全流程函数
# ================================================================
run_abl_t5() {
    local LABEL=$1
    local RQVAE_NAME=$2
    local OUTPUT_DIR=$3
    local GPUS=$4
    local PORT=$5
    shift 5
    local EXTRA_ARGS="$@"

    local INDEX_FILE=".index_lemb_${RQVAE_NAME}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${RQVAE_NAME}.json"

    echo "[${LABEL}] T5训练 GPU=${GPUS} port=${PORT} $(date)"
    mkdir -p $OUTPUT_DIR

    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
        --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
        --base_model ./config/ckpt \
        --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
        --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
        --max_his_len 20 --prompt_num 4 --patient 10 \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --tasks $TASKS --valid_task seqrec \
        $EXTRA_ARGS 2>&1 | tee $OUTPUT_DIR/train.log

    if [ ! -f "$OUTPUT_DIR/model.safetensors" ] && [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "[${LABEL}] T5 训练失败！"; return 1
    fi

    echo "[${LABEL}] seqrec推理 $(date)"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

    echo "[${LABEL}] seqimage推理 $(date)"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

    echo "[${LABEL}] ensemble $(date)"
    python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET --data_path ./data/ \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams $NUM_BEAMS

    echo "[${LABEL}] 完成 $(date)"
    cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null
    echo ""
}

# ================================================================
# 并行启动消融1 (GPU 0,1) 和消融2 (GPU 2,3)
# ================================================================
echo ""
echo "================================================================"
echo " 并行启动消融1 + 消融2"
echo " 消融1 (w/o concat): GPU 0,1 port 29660"
echo " 消融2 (w/o CAQ):    GPU 2,3 port 29662"
echo " $(date)"
echo "================================================================"

# 消融1: w/o concat (LLaMA+CAQ RQVAE + LCPA T5) — GPU 0,1
run_abl_t5 "w/o concat" $WO_CONCAT_RQVAE \
    "${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-b${NUM_BEAMS}" \
    "0,1" 29660 \
    --cpa_weight 0.005 --collab_emb_path $COLLAB --cpa_loss_type cosine &
PID_ABL1=$!

# 消融2: w/o CAQ (LLaMA+concat RQVAE + LCPA T5) — GPU 2,3
run_abl_t5 "w/o CAQ" $WO_CAQ_RQVAE \
    "${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-b${NUM_BEAMS}" \
    "2,3" 29662 \
    --cpa_weight 0.005 --collab_emb_path $COLLAB --cpa_loss_type cosine &
PID_ABL2=$!

echo "消融1 PID: $PID_ABL1 | 消融2 PID: $PID_ABL2"
wait $PID_ABL1
echo "消融1 (w/o concat) 完成"
wait $PID_ABL2
echo "消融2 (w/o CAQ) 完成"

echo ""
echo "========================================================"
echo " 消融1 + 消融2 全部完成！$(date)"
echo "========================================================"
