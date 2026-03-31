#!/bin/bash
# RQ3 cross_attn: T5 训练 + 推理 + Ensemble (RQVAE+索引已完成)
# GPU: 4,5

eval "$(conda shell.bash hook)"
conda activate ETEGRec
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
DATA_DIR="${BASE_DIR}/data/${DATASET}"
COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50
CPA_W=0.005
CPA_LOSS=cosine
TRAIN_GPUS="4,5"
PORT=29710

EXP_TAG="rq3-cross_attn"
INDEX_FILE=".index_lemb_${EXP_TAG}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_TAG}.json"
OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-${EXP_TAG}-b${NUM_BEAMS}"

cd $BASE_DIR

echo "================================================================"
echo " RQ3 cross_attn T5 | GPU: ${TRAIN_GPUS} | port: ${PORT}"
echo " 输出: ${OUTPUT_DIR}"
echo " 开始: $(date)"
echo "================================================================"

mkdir -p $OUTPUT_DIR

# T5 训练
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT \
    finetune_contrastive.py \
    --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
    --base_model ./config/ckpt \
    --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
    --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
    --max_his_len 20 --prompt_num 4 --patient 10 \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --tasks $TASKS --valid_task seqrec \
    --cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type $CPA_LOSS \
    2>&1 | tee $OUTPUT_DIR/train.log

if [ ! -f "$OUTPUT_DIR/model.safetensors" ] && [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
    echo "[${EXP_TAG}] T5 训练失败!"; exit 1
fi

# 推理 seqrec
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT \
    test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

# 推理 seqimage
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT \
    test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

# Ensemble
python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET \
    --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --num_beams $NUM_BEAMS

echo ""
echo "================================================================"
echo " [${EXP_TAG}] 全部完成! $(date)"
echo " 结果:"
cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null
echo "================================================================"
