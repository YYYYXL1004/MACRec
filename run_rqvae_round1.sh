#!/bin/bash
# Round 1: 6 组 RQVAE 并行对比实验 (Instruments)
# 3 张 GPU, 每张串行跑 2 组
# 用法: screen -dmS rqvae_r1 bash run_rqvae_round1.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled

DATASET="Instruments"
DATA_DIR="/data/yaoxianglin/ETEGRec/MACRec/data/${DATASET}"

TEXT_LLAMA="${DATA_DIR}/${DATASET}.emb-llama-td.npy"
TEXT_ST="${DATA_DIR}/${DATASET}.emb-st-768.npy"
IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

# RQVAE 通用参数
COMMON_ARGS="--num_emb_list 256 256 256 256 \
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
    --epochs 1000"

cd /data/yaoxianglin/ETEGRec/MACRec/cross_index

echo "============================================"
echo " Round 1: 6 组 RQVAE 并行对比 (Instruments)"
echo " 开始时间: $(date)"
echo "============================================"

# ========================================
# GPU 1: E1 + E2 (ST text, 拼接, 有/无 CAQ)
# ========================================
(
    # E1: ST + concat + CAQ
    CKPT_DIR="./log/${DATASET}/R1_E1_st-concat-caq"
    mkdir -p $CKPT_DIR
    echo "[E1] ST+concat+CAQ 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:1 \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} \
        --collab_fusion concat \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight 2.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E1] 完成 $(date)" | tee -a $CKPT_DIR/exp.log

    # E2: ST + concat + 无CAQ
    CKPT_DIR="./log/${DATASET}/R1_E2_st-concat-nocaq"
    mkdir -p $CKPT_DIR
    echo "[E2] ST+concat+无CAQ 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:1 \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} \
        --collab_fusion concat \
        --collab_contrastive_weight 0.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E2] 完成 $(date)" | tee -a $CKPT_DIR/exp.log
) &

# ========================================
# GPU 2: E3 + E4 (LLaMA text, 拼接/投影加法, 均有CAQ)
# ========================================
(
    # E3: LLaMA + concat + CAQ
    CKPT_DIR="./log/${DATASET}/R1_E3_llama-concat-caq"
    mkdir -p $CKPT_DIR
    echo "[E3] LLaMA+concat+CAQ 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:2 \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} \
        --collab_fusion concat \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight 2.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E3] 完成 $(date)" | tee -a $CKPT_DIR/exp.log

    # E4: LLaMA + proj_add + CAQ
    CKPT_DIR="./log/${DATASET}/R1_E4_llama-projadd-caq"
    mkdir -p $CKPT_DIR
    echo "[E4] LLaMA+投影加法+CAQ 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:2 \
        --text_data_path ${TEXT_LLAMA} \
        --image_data_path ${IMAGE} \
        --collab_data_path ${COLLAB} \
        --collab_fusion proj_add \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight 2.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E4] 完成 $(date)" | tee -a $CKPT_DIR/exp.log
) &

# ========================================
# GPU 3: E5 + E6 (ST text, 无collab / 无collab无CAQ baseline)
# ========================================
(
    # E5: ST + 无collab + CAQ
    CKPT_DIR="./log/${DATASET}/R1_E5_st-nocol-caq"
    mkdir -p $CKPT_DIR
    echo "[E5] ST+无collab+CAQ 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:3 \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --collab_neighbor_info ${COLLAB_NEIGHBOR} \
        --collab_contrastive_weight 2.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E5] 完成 $(date)" | tee -a $CKPT_DIR/exp.log

    # E6: ST + 无collab + 无CAQ (纯ST baseline)
    CKPT_DIR="./log/${DATASET}/R1_E6_st-baseline"
    mkdir -p $CKPT_DIR
    echo "[E6] ST baseline 开始 $(date)" | tee $CKPT_DIR/exp.log
    python -u main.py \
        --device cuda:3 \
        --text_data_path ${TEXT_ST} \
        --image_data_path ${IMAGE} \
        --collab_contrastive_weight 0.0 \
        --ckpt_dir $CKPT_DIR \
        $COMMON_ARGS 2>&1 | tee -a $CKPT_DIR/exp.log
    echo "[E6] 完成 $(date)" | tee -a $CKPT_DIR/exp.log
) &

# 等待所有 3 个 GPU 组完成
wait

echo ""
echo "============================================"
echo " Round 1 全部完成！$(date)"
echo "============================================"

# 汇总碰撞率
echo ""
echo "=== 碰撞率汇总 ==="
for dir in ./log/${DATASET}/R1_*/; do
    name=$(basename $dir)
    if [ -f "$dir/exp.log" ]; then
        text_col=$(grep -i "best text collision" $dir/exp.log | tail -1)
        img_col=$(grep -i "best image collision" $dir/exp.log | tail -1)
        echo "$name: $text_col | $img_col"
    fi
done
