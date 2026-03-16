#!/bin/bash
# eval_full.sh: 合并 Step4 的三步 (seqrec推理 + seqimage推理 + ensemble)
#
# 用法:
#   CUDA_VISIBLE_DEVICES=3,4 bash scripts/eval_full.sh <port> <save_name> <dataset>
#
# 示例:
#   CUDA_VISIBLE_DEVICES=3,4 bash scripts/eval_full.sh 29500 collab-3.15 Instruments
#
# 参数说明:
#   $1 - master_port (torchrun 端口, 如 29500)
#   $2 - save_name (模型名称后缀, 如 collab-3.15)
#   $3 - 数据集名称 (如 Instruments / Arts / Games)

set -e

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

port=${1:?请传入 master_port}
save_name=${2:?请传入 save_name}
Datasets=${3:?请传入数据集名称}

Index_file=.index_lemb_$save_name.json
Image_index_file=.index_vitemb_$save_name.json
OUTPUT_DIR=./log/$Datasets-$save_name-contrastive-align-0.01-temp-0.07-full-second
log_file=$OUTPUT_DIR/eval.log

echo "============================================"
echo " 评估配置"
echo " 数据集: $Datasets"
echo " 模型名: $save_name"
echo " 输出目录: $OUTPUT_DIR"
echo "============================================"

# 检查模型目录是否存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[ERROR] 模型目录不存在: $OUTPUT_DIR"
    exit 1
fi

# ---- Step 4a: 文本路推理 (seqrec) ----
echo ""
echo "[Step 4a] 文本路推理 (seqrec)..."
Valid_task=seqrec
results_file=$OUTPUT_DIR/results_${Valid_task}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task}_20.json

torchrun --nproc_per_node=2 --master_port=$port test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items 2>&1 | tee -a $log_file

echo "[Step 4a] seqrec 完成。结果: $results_file"

# ---- Step 4b: 图像路推理 (seqimage) ----
echo ""
echo "[Step 4b] 图像路推理 (seqimage)..."
Valid_task=seqimage
results_file=$OUTPUT_DIR/results_${Valid_task}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task}_20.json

torchrun --nproc_per_node=2 --master_port=$port test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items 2>&1 | tee -a $log_file

echo "[Step 4b] seqimage 完成。结果: $results_file"

# ---- Step 4c: 双路 Ensemble ----
echo ""
echo "[Step 4c] 双路 Ensemble..."
python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $Datasets \
    --data_path ./data/ \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --num_beams 20 2>&1 | tee -a $log_file

echo "[Step 4c] Ensemble 完成。结果: $OUTPUT_DIR/results_ensemble_20.json"

echo ""
echo "============================================"
echo " 所有评估步骤完成"
echo " seqrec:    $OUTPUT_DIR/results_seqrec_20.json"
echo " seqimage:  $OUTPUT_DIR/results_seqimage_20.json"
echo " ensemble:  $OUTPUT_DIR/results_ensemble_20.json"
echo " 日志:      $log_file"
echo "============================================"
