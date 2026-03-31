#!/bin/bash
# RQ4(a)(c): RQVAE 超参敏感性分析
# (a) CAR weight λ_r ∈ {0.5, 1.0, 5.0} (2.0 已完成)
# (c) align weight λ_a ∈ {0.0001, 0.01} (0.001 已完成)
# 每组: RQVAE 训练 → 索引生成 → T5 训练 → 推理 → Ensemble
# 用法: bash run_rq4_rqvae_hparams.sh <RQVAE_GPU> <TRAIN_GPUS> <PORT>

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
RQVAE_GPU="${1:-0}"
TRAIN_GPUS="${2:-0,1}"
PORT="${3:-29672}"

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

cd $BASE_DIR

# 完整的 RQVAE → T5 流程函数
run_full_pipeline() {
    local EXP_TAG=$1
    local CAR_W=$2
    local ALIGN_W=$3

    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/rq4-${EXP_TAG}"
    local INDEX_FILE=".index_lemb_rq4-${EXP_TAG}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_rq4-${EXP_TAG}.json"
    local OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-rq4-${EXP_TAG}-b${NUM_BEAMS}"

    echo ""
    echo "================================================================"
    echo " ${EXP_TAG}: λ_r=${CAR_W}, λ_a=${ALIGN_W}"
    echo " RQVAE: ${RQVAE_DIR} | T5: ${OUTPUT_DIR}"
    echo " 开始: $(date)"
    echo "================================================================"

    # Phase 0: RQVAE 训练
    mkdir -p $RQVAE_DIR
    cd ${BASE_DIR}/cross_index

    CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u main.py \
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
        echo " [${EXP_TAG}] RQVAE 训练失败，跳过"
        cd $BASE_DIR
        return 1
    fi

    # Phase 1: 索引生成
    CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
        --dataset $DATASET --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        --device cuda:0 --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
        --output_dir ${DATA_DIR} --output_file ${DATASET}${INDEX_FILE} --content text

    CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u generate_indices_distance.py \
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

    cd $BASE_DIR

    # Phase 2: T5 训练
    mkdir -p $OUTPUT_DIR
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
        --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
        --base_model ./config/ckpt \
        --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
        --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
        --max_his_len 20 --prompt_num 4 --patient 10 \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --tasks $TASKS --valid_task seqrec \
        --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS \
        2>&1 | tee $OUTPUT_DIR/train.log

    # Phase 3: 推理 + Ensemble
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqrec \
        --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
        --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
        --test_batch_size 64 --num_beams $NUM_BEAMS \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --test_task seqimage \
        --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
        --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

    python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET \
        --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $NUM_BEAMS

    echo " [${EXP_TAG}] 完成 $(date)"
    echo " 结果: $(cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null)"
}

echo "============================================"
echo " RQ4: RQVAE 超参敏感性分析"
echo " RQVAE GPU: ${RQVAE_GPU} | T5 GPU: ${TRAIN_GPUS}"
echo " 开始: $(date)"
echo "============================================"

# (a) λ_r 敏感性 (固定 λ_a=0.001)
run_full_pipeline "car0.5"  0.5  0.001
run_full_pipeline "car1.0"  1.0  0.001
run_full_pipeline "car5.0"  5.0  0.001

# (c) λ_a 敏感性 (固定 λ_r=2.0)
run_full_pipeline "align0.0001" 2.0 0.0001
run_full_pipeline "align0.01"   2.0 0.01

echo ""
echo "============================================"
echo " RQ4 RQVAE 超参实验全部完成 $(date)"
echo "============================================"
