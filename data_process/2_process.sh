# 用法: bash 2_process.sh Arts
# 支持: Arts / Games / Instruments
DATASET=${1:-Arts}

python amazon18_data_process.py --dataset $DATASET
