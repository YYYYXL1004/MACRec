#!/bin/bash
# Arts sc05 T5训练+推理+Ensemble (RQVAE和codes已完成)
# 与 sc03 并行在 GPU 4,5 上

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Arts"
EXP_NAME="R3_llama-sc05-caq"
TRAIN_GPUS="4,5"
PORT=29652
OUTPUT_DIR="/data/yaoxianglin/MACRec/log/${DATASET}-${EXP_NAME}"
INDEX_FILE=".index_lemb_${EXP_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"

cd /data/yaoxianglin/MACRec

echo "============================================"
echo " Arts sc05 T5+推理+Ensemble (并行)"
echo " GPU ${TRAIN_GPUS} | $(date)"
echo "============================================"

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
    --tasks seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image \
    --valid_task seqrec 2>&1 | tee $OUTPUT_DIR/train.log

echo "[T5完成] $(date)"

# seqrec 推理
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_20.json \
    --save_file $OUTPUT_DIR/save_seqrec_20.json \
    --filter_items

# seqimage 推理
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_20.json \
    --save_file $OUTPUT_DIR/save_seqimage_20.json \
    --filter_items

# Ensemble
python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams 20

echo ""
echo "============ sc05 结果 ============"
for f in results_seqrec_20.json results_seqimage_20.json results_ensemble_20.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
echo "[全部完成] $(date)"
