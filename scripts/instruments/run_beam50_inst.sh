#!/bin/bash
# Instruments beam-50 推理 + ensemble (基于 LCPA w=0.005 最佳模型)
# GPU 2,3

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
EXP_BASE="R1_E1_st-concat-caq"
OUTPUT_DIR="/data/yaoxianglin/MACRec/log/${DATASET}-lcpa-cosine-w0.005"

INDEX_FILE=".index_lemb_${EXP_BASE}.json"
IMAGE_INDEX_FILE=".index_vitemb_${EXP_BASE}.json"

TRAIN_GPUS="2,3"
PORT=29640
BEAM=50

cd /data/yaoxianglin/MACRec

echo "============================================"
echo " Beam-${BEAM} 推理: Instruments LCPA w=0.005"
echo " 开始时间: $(date)"
echo "============================================"

# 推理 seqrec beam-50
echo "[1/3] seqrec beam-${BEAM}... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 32 \
    --num_beams $BEAM \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqrec \
    --results_file $OUTPUT_DIR/results_seqrec_${BEAM}.json \
    --save_file $OUTPUT_DIR/save_seqrec_${BEAM}.json \
    --filter_items

# 推理 seqimage beam-50
echo "[2/3] seqimage beam-${BEAM}... $(date)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $DATASET \
    --test_batch_size 32 \
    --num_beams $BEAM \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --test_task seqimage \
    --results_file $OUTPUT_DIR/results_seqimage_${BEAM}.json \
    --save_file $OUTPUT_DIR/save_seqimage_${BEAM}.json \
    --filter_items

# Ensemble beam-50
echo "[3/3] Ensemble beam-${BEAM}... $(date)"
python ensemble.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ./data/ \
    --index_file $INDEX_FILE \
    --image_index_file $IMAGE_INDEX_FILE \
    --num_beams $BEAM

echo "[完成] $(date)"
echo "============ Beam-${BEAM} 结果 ============"
for f in results_seqrec_${BEAM}.json results_seqimage_${BEAM}.json results_ensemble_${BEAM}.json; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo "[$f]:"
        cat "$OUTPUT_DIR/$f"
        echo ""
    fi
done
