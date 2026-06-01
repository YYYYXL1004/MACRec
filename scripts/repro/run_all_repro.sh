#!/bin/bash
# One-shot: reproduce BOTH UNGER and MSCGRec for one dataset on one GPU.
# Regenerates all codes from scratch, then trains+evaluates both.
# Usage: bash scripts/repro/run_all_repro.sh <dataset> <gpu_id> <port>
#   e.g. bash scripts/repro/run_all_repro.sh Instruments 0 29531
# Run 3 datasets in parallel on 3 GPUs with distinct ports:
#   bash scripts/repro/run_all_repro.sh Instruments 0 29531 &
#   bash scripts/repro/run_all_repro.sh Arts        1 29532 &
#   bash scripts/repro/run_all_repro.sh Games       2 29533 &
set -e
DATASET=$1
GPU=$2
PORT=$3
PROJECT_DIR=/sda/data/yaoxianglin/MACRec
cd $PROJECT_DIR

echo "########## [$DATASET] UNGER: Stage-I unified code ##########"
cd analysis/repro
CUDA_VISIBLE_DEVICES=$GPU python unger_stage1.py --dataset $DATASET --data_root ../../data --epochs 200
cd $PROJECT_DIR

echo "########## [$DATASET] UNGER: T5 train + eval ##########"
bash scripts/repro/run_unger.sh $GPU $DATASET $PORT

echo "########## [$DATASET] MSCGRec: codes + T5 train + eval ##########"
bash scripts/repro/run_mscgrec.sh $GPU $DATASET $PORT

echo "########## [$DATASET] ALL DONE ##########"
echo "UNGER  : log/repro/${DATASET}-unger/results_seqrec_20.json"
echo "MSCGRec: log/repro/${DATASET}-mscgrec/results_mscgrec_20.json"
