#!/bin/bash
# RQ4(d): Reranking strength β 敏感性分析
# β ∈ {0, 0.05, 0.1, 0.2}
# 纯 CPU 后处理，不需要 GPU，使用已有最佳模型的 beam-50 候选
# 前置条件: save_seqrec_50.json 和 save_seqimage_50.json 已存在
# 用法: bash scripts/instruments/run_rq4_beta.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
BEST_DIR="${BASE_DIR}/log/${DATASET}-E3v2_llama-concat-caq-lcpa0.005-b50"
INDEX_FILE=".index_lemb_E3v2_llama-concat-caq.json"
IMAGE_INDEX_FILE=".index_vitemb_E3v2_llama-concat-caq.json"

# 检查前置文件
for f in save_seqrec_50.json save_seqimage_50.json; do
    if [ ! -f "${BEST_DIR}/${f}" ]; then
        echo "缺少 ${f}，需要先完成 beam-50 推理"
        exit 1
    fi
done

echo "============================================"
echo " RQ4(d): β 敏感性分析"
echo " β ∈ {0, 0.05, 0.1, 0.2}"
echo " 模型: ${BEST_DIR}"
echo " 开始: $(date)"
echo "============================================"

cd $BASE_DIR

PYTHONPATH=$BASE_DIR:$PYTHONPATH python scripts/rerank/rerank_collab.py \
    --data_path ./data/ \
    --dataset $DATASET \
    --output_dir $BEST_DIR \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams 50 \
    --beta "0,0.05,0.1,0.2" \
    --strategies "mean" \
    --combines "add"

echo ""
echo "============================================"
echo " RQ4(d): β 敏感性分析完成 $(date)"
echo "============================================"
