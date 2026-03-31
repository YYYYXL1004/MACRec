#!/bin/bash
# 消融4: w/o beam-50 (复用E3v2 LCPA模型, beam-20推理)
# GPU: 2,3

eval "$(conda shell.bash hook)"
conda activate ETEGRec
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
TRAIN_GPUS="2,3"
PORT=29662
BASE_DIR="/sda/data/yaoxianglin/MACRec"
NUM_BEAMS=20

E3V2_RQVAE="E3v2_llama-concat-caq"
INDEX_FILE=".index_lemb_${E3V2_RQVAE}.json"
IMAGE_INDEX_FILE=".index_vitemb_${E3V2_RQVAE}.json"
E3V2_LCPA_DIR="${BASE_DIR}/log/${DATASET}-${E3V2_RQVAE}-lcpa0.005-b50"

cd $BASE_DIR

echo "============================================"
echo " [消融4] w/o beam-50 (beam-20) | GPU: ${TRAIN_GPUS} | $(date)"
echo " 复用 E3v2 LCPA 模型, beam-${NUM_BEAMS} 推理"
echo "============================================"

# seqrec
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $E3V2_LCPA_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $E3V2_LCPA_DIR/results_seqrec_${NUM_BEAMS}.json \
    --save_file $E3V2_LCPA_DIR/save_seqrec_${NUM_BEAMS}.json --filter_items

# seqimage
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $E3V2_LCPA_DIR --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams $NUM_BEAMS \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $E3V2_LCPA_DIR/results_seqimage_${NUM_BEAMS}.json \
    --save_file $E3V2_LCPA_DIR/save_seqimage_${NUM_BEAMS}.json --filter_items

# ensemble
python ensemble.py --output_dir $E3V2_LCPA_DIR --dataset $DATASET --data_path ./data/ \
    --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams $NUM_BEAMS

echo "============================================"
echo " [消融4] w/o beam-50 完成 $(date)"
echo "============================================"
cat $E3V2_LCPA_DIR/results_ensemble_${NUM_BEAMS}.json 2>/dev/null
