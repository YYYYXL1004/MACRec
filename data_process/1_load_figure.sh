# 用法: bash 1_load_figure.sh Arts
# 支持: Arts / Games / Instruments
DATASET=${1:-Arts}

python load_all_figures.py --dataset $DATASET
