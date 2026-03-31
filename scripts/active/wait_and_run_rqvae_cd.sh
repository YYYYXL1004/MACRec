#!/bin/bash
# 等待 aug1/aug2 完成后启动 R-C 和 R-D（GPU 0,1 上并行）
# 用法: screen -dmS rq_cd_wait bash scripts/active/wait_and_run_rqvae_cd.sh

echo "等待增强实验 (aug1, aug2) 完成后启动 R-C + R-D..."
echo "开始监控时间: $(date)"

while true; do
    AUG1_ALIVE=$(screen -ls | grep -c "aug1")
    AUG2_ALIVE=$(screen -ls | grep -c "aug2")

    if [ "$AUG1_ALIVE" -eq 0 ] && [ "$AUG2_ALIVE" -eq 0 ]; then
        echo "增强实验已全部完成！$(date)"
        echo "等待 30 秒确保 GPU 完全释放..."
        sleep 30
        break
    fi

    echo "[$(date)] aug1=${AUG1_ALIVE} aug2=${AUG2_ALIVE}，继续等待..."
    sleep 300
done

echo "启动 R-C 和 R-D（GPU 0,1 并行）... $(date)"
cd /data/yaoxianglin/MACRec
screen -dmS rq_rc bash scripts/active/run_rqvae_RC.sh
screen -dmS rq_rd bash scripts/active/run_rqvae_RD.sh
echo "R-C 和 R-D 已启动"
screen -ls | grep "rq_r"
