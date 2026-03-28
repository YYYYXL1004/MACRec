#!/bin/bash
# 只跑 Step 2: RQVAE 空间检索质量对比 (嵌入文件已有)
# 用法: screen -dmS rqvae_cmp bash run_step2_only.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec
cd /data/yaoxianglin/MACRec

LOG_FILE="log/rqvae_retrieval_compare_results.txt"

echo "开始时间: $(date)" | tee $LOG_FILE

python -u rqvae_retrieval_compare.py \
    --dataset Instruments \
    --data_path ./data/ \
    --cag_text_emb ./data/Instruments/Instruments.emb-text-aligned-1024.npy \
    --cag_image_emb ./data/Instruments/Instruments.emb-image-aligned-1024.npy \
    --cag_output_dir log/Instruments-R1_E1_st-concat-caq \
    --cag_index_file .index_lemb_R1_E1_st-concat-caq.json \
    --cag_image_index_file .index_vitemb_R1_E1_st-concat-caq.json \
    --mac_output_dir log/Instruments-baseline \
    --mac_index_file .index_lemb_baseline.json \
    --mac_image_index_file .index_vitemb_baseline.json \
    --num_beams 20 \
    --collab_topk "10,20,50" \
    --collab_weight "0.05,0.1,0.2,0.5" 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
echo "结果已保存到: $LOG_FILE"
