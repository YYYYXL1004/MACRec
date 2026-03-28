#!/bin/bash
# 路线C实验：对比 CAGRec vs MACRec 的 RQVAE 空间检索质量
# 用法: screen -dmS rqvae_cmp bash run_rqvae_retrieval_compare.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

cd /data/yaoxianglin/MACRec
export CUDA_VISIBLE_DEVICES=1

echo "============================================"
echo " RQVAE 空间检索质量对比实验"
echo " 开始时间: $(date)"
echo "============================================"

DATASET="Instruments"
DATA_DIR="./data/${DATASET}"

# ================================================================
# Step 1: 提取 MACRec baseline 的 RQVAE 重建嵌入
# (CAGRec E1 的已有: emb-text-aligned-1024.npy, emb-image-aligned-1024.npy)
# ================================================================
echo ""
echo "[Step 1] 提取 MACRec baseline RQVAE 重建嵌入..."

# MACRec baseline: text path (LLaMA 4096d)
python -u extract_semantic_emb.py \
    --rqvae_ckpt cross_index/log/${DATASET}/baseline/best_text_collision_model.pth \
    --text_data_path ${DATA_DIR}/${DATASET}.emb-llama-td.npy \
    --image_data_path ${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy \
    --output_dir ${DATA_DIR} \
    --dataset ${DATASET}_baseline \
    --device cuda:0

# 重命名避免和 CAGRec 的混淆
if [ -f "${DATA_DIR}/${DATASET}_baseline.emb-text-aligned-4096.npy" ]; then
    echo "  MACRec baseline text aligned: 4096d ✓"
else
    echo "  Warning: MACRec baseline text aligned 未生成，检查输出维度..."
    ls ${DATA_DIR}/${DATASET}_baseline.emb-*aligned*.npy 2>/dev/null
fi

echo "[Step 1] 完成 $(date)"

# ================================================================
# Step 2: 运行通用检索对比脚本
# ================================================================
echo ""
echo "[Step 2] 运行检索质量对比..."

python -u rqvae_retrieval_compare.py \
    --dataset ${DATASET} \
    --data_path ./data/ \
    --cag_text_emb ${DATA_DIR}/${DATASET}.emb-text-aligned-1024.npy \
    --cag_image_emb ${DATA_DIR}/${DATASET}.emb-image-aligned-1024.npy \
    --cag_output_dir log/Instruments-R1_E1_st-concat-caq \
    --cag_index_file .index_lemb_R1_E1_st-concat-caq.json \
    --cag_image_index_file .index_vitemb_R1_E1_st-concat-caq.json \
    --mac_output_dir log/Instruments-baseline \
    --mac_index_file .index_lemb_baseline.json \
    --mac_image_index_file .index_vitemb_baseline.json \
    --num_beams 20 \
    --collab_topk "10,20,50" \
    --collab_weight "0.05,0.1,0.2,0.5"

echo ""
echo "============================================"
echo " 全部完成！$(date)"
echo "============================================"
