#!/bin/bash
# RQ3: Collaborative Embedding Fusion Strategy 对比
# No Collab / proj_add / gating / cross_attn (Concat 已在 E3v2 完成)
# 每组: RQVAE 训练 → 索引生成 → T5(+LCPA) 训练 → 推理 → Ensemble
# 用法: bash run_rq3_fusion.sh <RQVAE_GPU> <TRAIN_GPUS> <PORT>

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
RQVAE_GPU="${1:-0}"
TRAIN_GPUS="${2:-0,1}"
PORT="${3:-29674}"

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
CAR_W=2.0

cd $BASE_DIR

# 完整流程函数
run_fusion_pipeline() {
    local FUSION_TYPE=$1    # concat / proj_add / gating / cross_attn / none
    local USE_COLLAB=$2     # 1=有collab, 0=无collab
    local EXP_TAG="rq3-${FUSION_TYPE}"

    local RQVAE_DIR="${BASE_DIR}/cross_index/log/${DATASET}/${EXP_TAG}"
    local INDEX_FILE=".index_lemb_${EXP_TAG}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${EXP_TAG}.json"
    local OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-${EXP_TAG}-b${NUM_BEAMS}"

    echo ""
    echo "================================================================"
    echo " RQ3: fusion=${FUSION_TYPE}, collab=${USE_COLLAB}"
    echo " RQVAE: ${RQVAE_DIR} | T5: ${OUTPUT_DIR}"
    echo " 开始: $(date)"
    echo "================================================================"

    # 构建 RQVAE 参数
    local COLLAB_ARGS=""
    if [ "$USE_COLLAB" -eq 1 ]; then
        COLLAB_ARGS="--collab_data_path ${COLLAB} --collab_fusion ${FUSION_TYPE}"
        COLLAB_ARGS="${COLLAB_ARGS} --collab_neighbor_info ${COLLAB_NEIGHBOR} --collab_contrastive_weight ${CAR_W}"
    fi

    # Phase 0: RQVAE 训练
    mkdir -p $RQVAE_DIR
    cd ${BASE_DIR}/cross_index

    CUDA_VISIBLE_DEVICES=${RQVAE_GPU} python -u main.py \
        --device cuda:0 \
        --text_data_path ${TEXT_LLAMA} --image_data_path ${IMAGE} \
        ${COLLAB_ARGS} \
        --ckpt_dir $RQVAE_DIR \
        --num_emb_list 256 256 256 256 --e_dim 32 \
        --sk_epsilons 0.0 0.0 0.0 0.0 --eval_step 2 --batch_size 2048 \
        --begin_cross_layer 2 --use_cross_rq True \
        --text_class_info ${TEXT_CLASS} --image_class_info ${IMAGE_CLASS} \
        --text_contrast_weight 0.1 --image_contrast_weight 0.1 \
        --recon_contrast_weight 0.001 \
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

    # Phase 2: T5 训练 (所有 fusion 策略都配 LCPA)
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
echo " RQ3: Collaborative Embedding Fusion 对比"
echo " RQVAE GPU: ${RQVAE_GPU} | T5 GPU: ${TRAIN_GPUS}"
echo " 开始: $(date)"
echo "============================================"

# (i) No Collab: 跳过 (可后续单独跑)
# run_fusion_pipeline "no-collab"  0

# (ii) Addition: 投影加法
run_fusion_pipeline "proj_add"   1

# (iii) Gating: 门控融合
run_fusion_pipeline "gating"     1

# (iv) Cross-Attention: 交叉注意力
run_fusion_pipeline "cross_attn" 1

# (v) Concat: 已在 E3v2 完成，无需重跑

echo ""
echo "============================================"
echo " RQ3 全部完成 $(date)"
echo " Concat 结果请参考 E3v2 实验"
echo "============================================"
