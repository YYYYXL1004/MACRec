#!/bin/bash
# RQ4(b): CPA weight λ_p 敏感性分析
# λ_p ∈ {0, 0.002, 0.01} (0.005 已在最佳配置中完成)
# 复用 E3v2 RQVAE 索引，只需重训 T5
# GPU: 用户指定双卡

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

DATASET="Instruments"
BASE_DIR="/sda/data/yaoxianglin/MACRec"
TRAIN_GPUS="${1:-2,3}"
PORT="${2:-29670}"

INDEX_FILE=".index_lemb_E3v2_llama-concat-caq.json"
IMAGE_INDEX_FILE=".index_vitemb_E3v2_llama-concat-caq.json"
COLLAB="${BASE_DIR}/data/${DATASET}/${DATASET}.emb-collab-256.npy"
TASKS='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
NUM_BEAMS=50

cd $BASE_DIR

run_t5_pipeline() {
    local CPA_W=$1
    local TAG=$2
    local OUTPUT_DIR="${BASE_DIR}/log/${DATASET}-rq4-cpa${CPA_W}-b${NUM_BEAMS}"

    echo ""
    echo "================================================================"
    echo " λ_p=${CPA_W} | 输出: ${OUTPUT_DIR}"
    echo " 开始: $(date)"
    echo "================================================================"
    mkdir -p $OUTPUT_DIR

    # CPA参数
    local CPA_ARGS=""
    if [ "$(echo "$CPA_W > 0" | bc -l)" -eq 1 ]; then
        CPA_ARGS="--cpa_weight $CPA_W --collab_emb_path $COLLAB --cpa_loss_type cosine"
    fi

    # T5 训练
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
        --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
        --base_model ./config/ckpt \
        --per_device_batch_size 1024 --learning_rate 1e-3 --epochs 200 \
        --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
        --max_his_len 20 --prompt_num 4 --patient 10 \
        --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --tasks $TASKS --valid_task seqrec \
        $CPA_ARGS 2>&1 | tee $OUTPUT_DIR/train.log

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
    python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET \
        --data_path ./data/ --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
        --num_beams $NUM_BEAMS

    echo " λ_p=${CPA_W} 完成 $(date)"
    echo " 结果: $(cat $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json)"
}

echo "============================================"
echo " RQ4(b): CPA weight λ_p 敏感性分析"
echo " λ_p ∈ {0, 0.002, 0.01}"
echo " GPU: ${TRAIN_GPUS} | Port: ${PORT}"
echo " 开始: $(date)"
echo "============================================"

run_t5_pipeline 0     "no-cpa"
run_t5_pipeline 0.002 "cpa0.002"
run_t5_pipeline 0.01  "cpa0.01"

echo ""
echo "============================================"
echo " RQ4(b) 全部完成 $(date)"
echo "============================================"
