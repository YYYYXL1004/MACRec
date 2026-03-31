#!/bin/bash
# Instruments w/o 消融实验 (基于最佳模型 E3v2 减组件)
# 最佳: LLaMA+concat+CAQ+LCPA+beam-50 → N@10=0.1070
# 消融1: w/o concat  (RQVAE: LLaMA+CAQ, T5: LCPA, beam-50)
# 消融2: w/o CAQ     (RQVAE: LLaMA+concat, T5: LCPA, beam-50)
# 消融3: w/o LCPA    (RQVAE: 复用E3v2, T5: 无LCPA, beam-50)
# 消融4: w/o beam-50 (复用E3v2模型, beam-20推理)
# GPU: 0,1

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
TRAIN_GPUS="0,1"
PORT=29660

BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"

# 数据文件
TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'

# E3v2 (最佳模型) 的 RQVAE 索引名
E3V2_RQVAE="E3v2_llama-concat-caq"
E3V2_LCPA_DIR="${BASE_DIR}/log/${DATASET}-${E3V2_RQVAE}-lcpa0.005-b50"

cd $BASE_DIR

echo "============================================"
echo " Instruments w/o 消融实验 (4组)"
echo " 最佳模型: E3v2 (LLaMA+concat+CAQ+LCPA+beam-50)"
echo " GPU: ${TRAIN_GPUS}"
echo " 开始时间: $(date)"
echo "============================================"

# ================================================================
# Phase 1: 并行训练两个 RQVAE (GPU 0 / GPU 1)
# ================================================================
echo ""
echo "================================================================"
echo " [Phase 1] 并行训练 RQVAE (GPU 0: w/o concat, GPU 1: w/o CAQ)"
echo " 开始: $(date)"
echo "================================================================"

# --- w/o concat: RQVAE = LLaMA + CAQ(w=2.0), 无collab concat ---
WO_CONCAT_RQVAE="wo-concat_llama-caq"
WO_CONCAT_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CONCAT_RQVAE}"
mkdir -p $WO_CONCAT_RQVAE_DIR

echo "[Phase 1a] w/o concat RQVAE 启动 (GPU 0)... $(date)"
cd ${BASE_DIR}/cross_index
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} \
    --image_data_path ${IMAGE} \
    --collab_neighbor_info ${COLLAB_NEIGHBOR} \
    --collab_contrastive_weight 2.0 \
    --ckpt_dir $WO_CONCAT_RQVAE_DIR \
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
    --epochs 1000 2>&1 | tee $WO_CONCAT_RQVAE_DIR/train.log &
PID_RQVAE1=$!

# --- w/o CAQ: RQVAE = LLaMA + concat, 无CAQ ---
WO_CAQ_RQVAE="wo-caq_llama-concat"
WO_CAQ_RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${WO_CAQ_RQVAE}"
mkdir -p $WO_CAQ_RQVAE_DIR

echo "[Phase 1b] w/o CAQ RQVAE 启动 (GPU 1)... $(date)"
CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --device cuda:0 \
    --text_data_path ${TEXT_LLAMA} \
    --image_data_path ${IMAGE} \
    --collab_data_path ${COLLAB} \
    --collab_fusion concat \
    --collab_contrastive_weight 0.0 \
    --ckpt_dir $WO_CAQ_RQVAE_DIR \
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
    --epochs 1000 2>&1 | tee $WO_CAQ_RQVAE_DIR/train.log &
PID_RQVAE2=$!

cd $BASE_DIR

echo "等待两个 RQVAE 训练完成... (PID: $PID_RQVAE1, $PID_RQVAE2)"
wait $PID_RQVAE1
echo "[Phase 1a] w/o concat RQVAE 完成 $(date)"
wait $PID_RQVAE2
echo "[Phase 1b] w/o CAQ RQVAE 完成 $(date)"

# 检查模型文件
for rdir in $WO_CONCAT_RQVAE_DIR $WO_CAQ_RQVAE_DIR; do
    if [ ! -f "$rdir/best_text_collision_model.pth" ] || [ ! -f "$rdir/best_image_collision_model.pth" ]; then
        echo "[Phase 1] RQVAE 训练失败: $rdir"
        exit 1
    fi
done
echo "[Phase 1] 两个 RQVAE 均训练完成 $(date)"

# ================================================================
# Phase 2: 生成索引
# ================================================================
echo ""
echo "================================================================"
echo " [Phase 2] 生成离散索引"
echo "================================================================"

generate_indices() {
    local EXP_NAME=$1
    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_NAME}"
    local IDX_FILE="${DATASET}.index_lemb_${EXP_NAME}.json"
    local IMG_IDX_FILE="${DATASET}.index_vitemb_${EXP_NAME}.json"

    cd ${BASE_DIR}/cross_index

    echo "  生成 ${EXP_NAME} text code..."
    CUDA_VISIBLE_DEVICES=0 python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${IDX_FILE} \
        --content text

    echo "  生成 ${EXP_NAME} image code..."
    CUDA_VISIBLE_DEVICES=0 python -u generate_indices_distance.py \
        --dataset $DATASET \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --device cuda:0 \
        --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
        --output_dir ${DATA_DIR} \
        --output_file ${IMG_IDX_FILE} \
        --content image

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

generate_indices $WO_CONCAT_RQVAE
generate_indices $WO_CAQ_RQVAE

echo "[Phase 2] 索引生成完成 $(date)"

# ================================================================
# 通用函数: T5训练 + 推理 + Ensemble
# ================================================================
run_t5_full() {
    local LABEL=$1
    local RQVAE_NAME=$2
    local OUTPUT_DIR=$3
    local NUM_BEAMS=$4
    shift 4
    local EXTRA_T5_ARGS="$@"

    local INDEX_FILE=".index_lemb_${RQVAE_NAME}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${RQVAE_NAME}.json"

    echo ""
    echo "------------------------------------------------------------"
    echo " [${LABEL}] T5 训练 → 推理 → Ensemble"
    echo " RQVAE索引: ${RQVAE_NAME} | beam: ${NUM_BEAMS}"
    echo " 输出: ${OUTPUT_DIR}"
    echo " 额外参数: ${EXTRA_T5_ARGS}"
    echo " 开始: $(date)"
    echo "------------------------------------------------------------"

    mkdir -p $OUTPUT_DIR

    # T5 训练
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
        --data_path ./data/ \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR \
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
        $EXTRA_T5_ARGS 2>&1 | tee $OUTPUT_DIR/train.log

    if [ ! -f "$OUTPUT_DIR/model.safetensors" ] && [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "[${LABEL}] T5 训练失败！"
        exit 1
    fi
    echo "[${LABEL}] T5 训练完成 $(date)"

    # 推理 + Ensemble
    run_inference $LABEL $OUTPUT_DIR $INDEX_FILE $IMAGE_INDEX_FILE $NUM_BEAMS
}

run_inference() {
    local LABEL=$1
    local OUTPUT_DIR=$2
    local INDEX_FILE=$3
    local IMAGE_INDEX_FILE=$4
    local NUM_BEAMS=$5

    echo "[${LABEL}] seqrec 推理 (beam-${NUM_BEAMS})... $(date)"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR \
        --data_path ./data/ \
        --dataset $DATASET \
        --test_batch_size 64 \
        --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json \
        --filter_items

    echo "[${LABEL}] seqimage 推理 (beam-${NUM_BEAMS})... $(date)"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR \
        --data_path ./data/ \
        --dataset $DATASET \
        --test_batch_size 64 \
        --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json \
        --filter_items

    echo "[${LABEL}] Ensemble (beam-${NUM_BEAMS})... $(date)"
    python ensemble.py \
        --output_dir $OUTPUT_DIR \
        --dataset $DATASET \
        --data_path ./data/ \
        --index_file $INDEX_FILE \
        --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $NUM_BEAMS

    echo ""
    echo "=== [${LABEL}] 结果 ==="
    for f in results_seqrec_${NUM_BEAMS}.json results_seqimage_${NUM_BEAMS}.json results_ensemble_${NUM_BEAMS}.json; do
        if [ -f "$OUTPUT_DIR/$f" ]; then
            echo "[$f]:"
            cat "$OUTPUT_DIR/$f"
            echo ""
        fi
    done
}

# ================================================================
# Phase 3: 消融1 — w/o concat (RQVAE: LLaMA+CAQ, T5: LCPA, beam-50)
# ================================================================
echo ""
echo "################################################################"
echo " [消融1] w/o concat"
echo "################################################################"

run_t5_full "w/o concat" $WO_CONCAT_RQVAE \
    "${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-b50" 50 \
    --cpa_weight 0.005 --collab_emb_path $COLLAB --cpa_loss_type cosine

# ================================================================
# Phase 4: 消融2 — w/o CAQ (RQVAE: LLaMA+concat, T5: LCPA, beam-50)
# ================================================================
echo ""
echo "################################################################"
echo " [消融2] w/o CAQ"
echo "################################################################"

run_t5_full "w/o CAQ" $WO_CAQ_RQVAE \
    "${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-b50" 50 \
    --cpa_weight 0.005 --collab_emb_path $COLLAB --cpa_loss_type cosine

# ================================================================
# Phase 5: 消融3 — w/o LCPA (复用E3v2 RQVAE, T5无LCPA, beam-50)
# ================================================================
echo ""
echo "################################################################"
echo " [消融3] w/o LCPA (复用E3v2 RQVAE索引)"
echo "################################################################"

WO_LCPA_OUTPUT="${BASE_DIR}/log/${DATASET}-wo-lcpa-b50"
INDEX_FILE_E3V2=".index_lemb_${E3V2_RQVAE}.json"
IMAGE_INDEX_FILE_E3V2=".index_vitemb_${E3V2_RQVAE}.json"

echo "[w/o LCPA] T5 训练 (无LCPA)... $(date)"
mkdir -p $WO_LCPA_OUTPUT

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $WO_LCPA_OUTPUT \
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
    --index_file $INDEX_FILE_E3V2 \
    --image_index_file $IMAGE_INDEX_FILE_E3V2 \
    --tasks $TASKS \
    --valid_task seqrec 2>&1 | tee $WO_LCPA_OUTPUT/train.log

if [ ! -f "$WO_LCPA_OUTPUT/model.safetensors" ] && [ ! -f "$WO_LCPA_OUTPUT/pytorch_model.bin" ]; then
    echo "[w/o LCPA] T5 训练失败！"
    exit 1
fi
echo "[w/o LCPA] T5 训练完成 $(date)"

run_inference "w/o LCPA" $WO_LCPA_OUTPUT $INDEX_FILE_E3V2 $IMAGE_INDEX_FILE_E3V2 50

# ================================================================
# Phase 6: 消融4 — w/o beam-50 (复用E3v2 LCPA模型, beam-20推理)
# ================================================================
echo ""
echo "################################################################"
echo " [消融4] w/o beam-50 (复用E3v2 LCPA模型, beam-20)"
echo "################################################################"

run_inference "w/o beam-50" $E3V2_LCPA_DIR $INDEX_FILE_E3V2 $IMAGE_INDEX_FILE_E3V2 20

# ================================================================
# 汇总
# ================================================================
echo ""
echo "========================================================"
echo " 四组消融实验全部完成！$(date)"
echo ""
echo " Full model (E3v2):  N@10=0.1070"
echo " w/o concat:  ${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-b50"
echo " w/o CAQ:     ${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-b50"
echo " w/o LCPA:    ${WO_LCPA_OUTPUT}"
echo " w/o beam-50: ${E3V2_LCPA_DIR} (beam-20 results)"
echo "========================================================"

echo ""
echo "=== Ensemble 结果汇总 ==="
echo "[Full model (E3v2+LCPA beam-50)]:"
cat "${E3V2_LCPA_DIR}/results_ensemble_50.json" 2>/dev/null
echo ""

for dir_label in "${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-b50|w/o concat (b50)" "${BASE_DIR}/log/${DATASET}-wo-caq-lcpa-b50|w/o CAQ (b50)" "${WO_LCPA_OUTPUT}|w/o LCPA (b50)" "${E3V2_LCPA_DIR}|w/o beam-50 (b20)"; do
    IFS='|' read -r dir label <<< "$dir_label"
    bsize=50
    if [[ "$label" == *"b20"* ]]; then bsize=20; fi
    if [ -f "$dir/results_ensemble_${bsize}.json" ]; then
        echo "[${label}]:"
        cat "$dir/results_ensemble_${bsize}.json"
        echo ""
    fi
done
