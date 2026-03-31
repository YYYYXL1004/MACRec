#!/bin/bash
# 消融1: w/o concat — T5 训练 + 推理 + ensemble (重跑, 修复端口冲突)
# RQVAE + 索引已完成, 仅跑 T5 阶段
# GPU: 0,1

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
TRAIN_GPUS="0,1"
PORT=29680

RQVAE_NAME="wo-concat_llama-caq"
INDEX_FILE=".index_lemb_${RQVAE_NAME}.json"
IMAGE_INDEX_FILE=".index_vitemb_${RQVAE_NAME}.json"
OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-wo-concat-lcpa-b${NUM_BEAMS}"

cd $BASE_DIR

echo "============================================"
echo " [消融1] w/o concat T5 重跑 | GPU: ${TRAIN_GPUS} | port: ${PORT}"
echo " $(date)"
echo "============================================"

# 清理之前失败的目录
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# T5 训练
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
    --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
    --base_model ./config/ckpt \
    --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
    --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
    --max_his_len 20 --prompt_num 4 --patient 10 \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --tasks $TASKS --valid_task seqrec \
    --cpa_weight 0.005 --collab_emb_path $COLLAB --cpa_loss_type cosine \
    2>&1 | tee $OUTPUT_DIR/train.log

if [ ! -f "$OUTPUT_DIR/model.safetensors" ] && [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
    echo "[消融1] T5 训练失败!"; exit 1
fi

# seqrec 推理
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

# seqimage 推理
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json \
    --save_file $OUTPUT_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

# Ensemble
python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET --data_path ./data/ \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams $NUM_BEAMS

echo "============================================"
echo " [消融1] w/o concat 完成! $(date)"
echo "============================================"
cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null
