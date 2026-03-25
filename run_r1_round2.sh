#!/bin/bash
# Round 1 扩展: E3(Inst) + Arts + Games 并行
# GPU 2,3: Arts + E3 (串行各一组全流程)
# GPU 1,6: Games
# 用法: screen -dmS r1_round2 bash run_r1_round2.sh

eval "$(conda shell.bash hook)"
conda activate ETEGRec

export WANDB_MODE=disabled
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

PROJECT="/data/yaoxianglin/ETEGRec/MACRec"

# ======================================================================
# 通用函数: 数据准备 (生成 ST embedding + collab embedding + neighbors)
# ======================================================================
prepare_data() {
    local DATASET=$1
    local GPU_ID=$2
    local DATA_DIR="${PROJECT}/data/${DATASET}"

    echo "[准备] ${DATASET} 数据准备开始 $(date)"

    # ST text embedding
    if [ ! -f "${DATA_DIR}/${DATASET}.emb-st-768.npy" ]; then
        echo "  生成 ST text embedding..."
        cd ${PROJECT}/data_process
        python -u gen_text_emb_st.py --dataset ${DATASET} --gpu_id ${GPU_ID} 2>&1 | tail -5
    else
        echo "  ST text embedding 已存在，跳过"
    fi

    # collab embedding (SASRec)
    if [ ! -f "${DATA_DIR}/${DATASET}.emb-collab-256.npy" ]; then
        echo "  生成 collab embedding (SASRec)..."
        cd ${PROJECT}/data_process
        python -u gen_collab_emb.py --dataset ${DATASET} --gpu_id ${GPU_ID} 2>&1 | tail -10
    else
        echo "  collab embedding 已存在，跳过"
    fi

    # collab neighbors
    if [ ! -f "${DATA_DIR}/${DATASET}.collab_neighbors_k10.json" ]; then
        echo "  生成 collab neighbors..."
        cd ${PROJECT}/cross_index
        python -u gen_collab_neighbors.py --dataset ${DATASET} --topk 10 2>&1 | tail -5
    else
        echo "  collab neighbors 已存在，跳过"
    fi

    # KMeans 伪标签 (如果用 ST 需要重新生成)
    # 当前已有 LLaMA 的 kmeans，ST 维度不同但伪标签只和内容相关，暂时沿用 LLaMA 版
    # 如需重生成:
    # cd ${PROJECT}/data && python kmeans.py --dataset ${DATASET} --text_emb_suffix .emb-st-768.npy

    echo "[准备] ${DATASET} 数据准备完成 $(date)"
}

# ======================================================================
# 通用函数: E1 配置全流程 (ST+concat+CAQ)
# ======================================================================
run_e1_pipeline() {
    local DATASET=$1
    local TRAIN_GPUS=$2
    local PORT=$3
    local CODE_GPU=$4   # code 生成用的单卡编号

    local EXP_NAME="R1_E1_st-concat-caq"
    local DATA_DIR="${PROJECT}/data/${DATASET}"
    local RQVAE_DIR="${PROJECT}/cross_index/log/${DATASET}/${EXP_NAME}"
    local OUTPUT_DIR="${PROJECT}/log/${DATASET}-${EXP_NAME}"
    local CHECKPOINT_DIR="${PROJECT}/.pipeline_${DATASET}_${EXP_NAME}"

    local TEXT_ST="${DATA_DIR}/${DATASET}.emb-st-768.npy"
    local IMAGE="${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy"
    local COLLAB="${DATA_DIR}/${DATASET}.emb-collab-256.npy"
    local COLLAB_NEIGHBOR="${DATA_DIR}/${DATASET}.collab_neighbors_k10.json"
    local TEXT_CLASS="${DATA_DIR}/${DATASET}.index_lemb_kmeans512.json"
    local IMAGE_CLASS="${DATA_DIR}/${DATASET}.index_vitemb_kmeans512.json"

    local INDEX_FILE=".index_lemb_${EXP_NAME}.json"
    local IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"

    mkdir -p $CHECKPOINT_DIR

    echo ""
    echo "============================================"
    echo " ${DATASET} ${EXP_NAME} 全流程"
    echo " GPU: ${TRAIN_GPUS}, 开始时间: $(date)"
    echo "============================================"

    # Step 1: RQVAE 训练
    if [ -f "$CHECKPOINT_DIR/rqvae_done" ]; then
        echo "[RQVAE] 已完成，跳过"
    else
        echo "[RQVAE] 训练 CrossRQVAE... $(date)"
        cd ${PROJECT}/cross_index
        mkdir -p $RQVAE_DIR

        python -u main.py \
            --device cuda:${CODE_GPU} \
            --text_data_path ${TEXT_ST} \
            --image_data_path ${IMAGE} \
            --collab_data_path ${COLLAB} \
            --collab_fusion concat \
            --collab_neighbor_info ${COLLAB_NEIGHBOR} \
            --collab_contrastive_weight 2.0 \
            --ckpt_dir $RQVAE_DIR \
            --num_emb_list 256 256 256 256 \
            --sk_epsilons 0.0 0.0 0.0 0.0 \
            --eval_step 2 \
            --batch_size 2048 \
            --begin_cross_layer 2 \
            --use_cross_rq True \
            --text_class_info ${TEXT_CLASS} \
            --image_class_info ${IMAGE_CLASS} \
            --text_contrast_weight 0.1 \
            --image_contrast_weight 0.1 \
            --recon_contrast_weight 0.001 \
            --epochs 1000 2>&1 | tee $RQVAE_DIR/train.log

        if [ -f "$RQVAE_DIR/best_text_collision_model.pth" ]; then
            touch "$CHECKPOINT_DIR/rqvae_done"
            echo "[RQVAE] 完成 $(date)"
        else
            echo "[RQVAE] 失败！"; return 1
        fi
    fi

    # Step 2: 生成 text code
    if [ -f "$CHECKPOINT_DIR/textcode_done" ]; then
        echo "[TextCode] 已完成，跳过"
    else
        echo "[TextCode] 生成 text index code... $(date)"
        cd ${PROJECT}/cross_index
        python -u generate_indices_distance.py \
            --dataset $DATASET \
            --text_data_path ${TEXT_ST} \
            --image_data_path ${IMAGE} \
            --device cuda:${CODE_GPU} \
            --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
            --output_dir ${DATA_DIR} \
            --output_file ${DATASET}${INDEX_FILE} \
            --content text
        if [ -f "${DATA_DIR}/${DATASET}${INDEX_FILE}" ]; then
            touch "$CHECKPOINT_DIR/textcode_done"
        else
            echo "[TextCode] 失败！"; return 1
        fi
    fi

    # Step 3: 生成 image code
    if [ -f "$CHECKPOINT_DIR/imgcode_done" ]; then
        echo "[ImgCode] 已完成，跳过"
    else
        echo "[ImgCode] 生成 image index code... $(date)"
        cd ${PROJECT}/cross_index
        python -u generate_indices_distance.py \
            --dataset $DATASET \
            --text_data_path ${TEXT_ST} \
            --image_data_path ${IMAGE} \
            --device cuda:${CODE_GPU} \
            --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
            --output_dir ${DATA_DIR} \
            --output_file ${DATASET}${IMAGE_INDEX_FILE} \
            --content image
        if [ -f "${DATA_DIR}/${DATASET}${IMAGE_INDEX_FILE}" ]; then
            touch "$CHECKPOINT_DIR/imgcode_done"
        else
            echo "[ImgCode] 失败！"; return 1
        fi
    fi

    # Step 4: T5 训练
    if [ -f "$CHECKPOINT_DIR/t5_done" ]; then
        echo "[T5] 已完成，跳过"
    else
        echo "[T5] 训练... $(date)"
        cd ${PROJECT}
        mkdir -p $OUTPUT_DIR

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

        if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]; then
            touch "$CHECKPOINT_DIR/t5_done"
        else
            echo "[T5] 失败！"; return 1
        fi
    fi

    # Step 5: 推理 seqrec
    if [ -f "$CHECKPOINT_DIR/seqrec_done" ]; then
        echo "[seqrec] 已完成，跳过"
    else
        echo "[seqrec] 推理... $(date)"
        cd ${PROJECT}
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
            --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
            --test_batch_size 64 --num_beams 20 \
            --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
            --test_task seqrec \
            --results_file $OUTPUT_DIR/results_seqrec_20.json \
            --save_file $OUTPUT_DIR/save_seqrec_20.json --filter_items
        [ -f "$OUTPUT_DIR/results_seqrec_20.json" ] && touch "$CHECKPOINT_DIR/seqrec_done"
    fi

    # Step 6: 推理 seqimage
    if [ -f "$CHECKPOINT_DIR/seqimage_done" ]; then
        echo "[seqimage] 已完成，跳过"
    else
        echo "[seqimage] 推理... $(date)"
        cd ${PROJECT}
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
            --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
            --test_batch_size 64 --num_beams 20 \
            --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
            --test_task seqimage \
            --results_file $OUTPUT_DIR/results_seqimage_20.json \
            --save_file $OUTPUT_DIR/save_seqimage_20.json --filter_items
        [ -f "$OUTPUT_DIR/results_seqimage_20.json" ] && touch "$CHECKPOINT_DIR/seqimage_done"
    fi

    # Step 7: Ensemble
    if [ -f "$CHECKPOINT_DIR/ensemble_done" ]; then
        echo "[Ensemble] 已完成，跳过"
    else
        echo "[Ensemble] ... $(date)"
        cd ${PROJECT}
        python ensemble.py \
            --output_dir $OUTPUT_DIR --dataset $DATASET --data_path ./data/ \
            --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams 20
        [ -f "$OUTPUT_DIR/results_ensemble_20.json" ] && touch "$CHECKPOINT_DIR/ensemble_done"
    fi

    echo ""
    echo "=== ${DATASET} 结果 ==="
    [ -f "$OUTPUT_DIR/results_ensemble_20.json" ] && cat "$OUTPUT_DIR/results_ensemble_20.json"
    echo ""
}

# ======================================================================
# GPU 2,3: E3 (Inst, LLaMA+concat+CAQ) → Arts (ST+concat+CAQ)
# ======================================================================
(
    # --- E3: Instruments LLaMA+concat+CAQ ---
    DATASET="Instruments"
    EXP_NAME="R1_E3_llama-concat-caq"
    RQVAE_DIR="${PROJECT}/cross_index/log/${DATASET}/${EXP_NAME}"
    OUTPUT_DIR="${PROJECT}/log/${DATASET}-${EXP_NAME}"
    CHECKPOINT_DIR="${PROJECT}/.pipeline_${DATASET}_${EXP_NAME}"
    DATA_DIR="${PROJECT}/data/${DATASET}"
    TRAIN_GPUS="2,3"
    PORT=29620
    INDEX_FILE=".index_lemb_${EXP_NAME}.json"
    IMAGE_INDEX_FILE=".index_vitemb_${EXP_NAME}.json"
    mkdir -p $CHECKPOINT_DIR

    echo "============================================"
    echo " E3: Inst LLaMA+concat+CAQ 全流程"
    echo "============================================"

    # RQVAE 已训练，只需生成 code + T5
    # 生成 text code
    if [ ! -f "$CHECKPOINT_DIR/textcode_done" ]; then
        cd ${PROJECT}/cross_index
        python -u generate_indices_distance.py \
            --dataset $DATASET \
            --text_data_path ${DATA_DIR}/${DATASET}.emb-llama-td.npy \
            --image_data_path ${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy \
            --device cuda:2 \
            --ckpt_path ${RQVAE_DIR}/best_text_collision_model.pth \
            --output_dir ${DATA_DIR} \
            --output_file ${DATASET}${INDEX_FILE} \
            --content text
        touch "$CHECKPOINT_DIR/textcode_done"
    fi

    # 生成 image code
    if [ ! -f "$CHECKPOINT_DIR/imgcode_done" ]; then
        cd ${PROJECT}/cross_index
        python -u generate_indices_distance.py \
            --dataset $DATASET \
            --text_data_path ${DATA_DIR}/${DATASET}.emb-llama-td.npy \
            --image_data_path ${DATA_DIR}/${DATASET}.emb-ViT-L-14.npy \
            --device cuda:2 \
            --ckpt_path ${RQVAE_DIR}/best_image_collision_model.pth \
            --output_dir ${DATA_DIR} \
            --output_file ${DATASET}${IMAGE_INDEX_FILE} \
            --content image
        touch "$CHECKPOINT_DIR/imgcode_done"
    fi

    # T5 训练
    if [ ! -f "$CHECKPOINT_DIR/t5_done" ]; then
        cd ${PROJECT}
        mkdir -p $OUTPUT_DIR
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT finetune_contrastive.py \
            --data_path ./data/ --dataset $DATASET --output_dir $OUTPUT_DIR \
            --base_model ./config/ckpt --per_device_batch_size 1024 --learning_rate 1e-3 \
            --epochs 200 --weight_decay 0.01 --save_and_eval_strategy epoch --logging_step 50 \
            --max_his_len 20 --prompt_num 4 --patient 10 \
            --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
            --tasks seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image \
            --valid_task seqrec 2>&1 | tee $OUTPUT_DIR/train.log
        ([ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]) && touch "$CHECKPOINT_DIR/t5_done"
    fi

    # 推理 + ensemble
    if [ ! -f "$CHECKPOINT_DIR/seqrec_done" ]; then
        cd ${PROJECT}
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
            --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
            --test_batch_size 64 --num_beams 20 --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
            --test_task seqrec --results_file $OUTPUT_DIR/results_seqrec_20.json \
            --save_file $OUTPUT_DIR/save_seqrec_20.json --filter_items
        touch "$CHECKPOINT_DIR/seqrec_done"
    fi
    if [ ! -f "$CHECKPOINT_DIR/seqimage_done" ]; then
        cd ${PROJECT}
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=2 --master_port=$PORT test_ddp_save.py \
            --ckpt_path $OUTPUT_DIR --data_path ./data/ --dataset $DATASET \
            --test_batch_size 64 --num_beams 20 --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE \
            --test_task seqimage --results_file $OUTPUT_DIR/results_seqimage_20.json \
            --save_file $OUTPUT_DIR/save_seqimage_20.json --filter_items
        touch "$CHECKPOINT_DIR/seqimage_done"
    fi
    if [ ! -f "$CHECKPOINT_DIR/ensemble_done" ]; then
        cd ${PROJECT}
        python ensemble.py --output_dir $OUTPUT_DIR --dataset $DATASET --data_path ./data/ \
            --index_file $INDEX_FILE --image_index_file $IMAGE_INDEX_FILE --num_beams 20
        touch "$CHECKPOINT_DIR/ensemble_done"
    fi
    echo "=== E3 Inst 结果 ===" && cat "$OUTPUT_DIR/results_ensemble_20.json" 2>/dev/null && echo ""

    # --- Arts E1 (ST+concat+CAQ) ---
    prepare_data "Arts" 2
    run_e1_pipeline "Arts" "2,3" 29620 2

) &

# ======================================================================
# GPU 1,6: Games E1 (ST+concat+CAQ)
# ======================================================================
(
    prepare_data "Games" 1
    run_e1_pipeline "Games" "1,6" 29621 1
) &

wait

echo ""
echo "============================================"
echo " 全部完成！$(date)"
echo "============================================"
echo ""
echo "=== 结果汇总 ==="
for dir in ${PROJECT}/log/*-R1_E*; do
    name=$(basename $dir)
    if [ -f "$dir/results_ensemble_20.json" ]; then
        ndcg=$(python3 -c "import json; d=json.load(open('$dir/results_ensemble_20.json')); print(f'{d[\"ndcg@10\"]:.4f}')")
        echo "$name: NDCG@10=$ndcg"
    fi
done
