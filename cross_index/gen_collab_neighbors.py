"""
从 SASRec 协同向量生成 item 的协同邻居表 (top-K nearest neighbors)

用法:
    cd MACRec/cross_index
    python gen_collab_neighbors.py --dataset Instruments --topk 10
"""

import argparse
import json
import os
import numpy as np
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--collab_suffix', type=str, default='.emb-collab-256.npy')
    parser.add_argument('--topk', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    K = args.topk

    # 加载并 L2 归一化
    collab_path = os.path.join(args.data_dir, ds, ds + args.collab_suffix)
    collab = np.load(collab_path).astype(np.float32)
    collab_norm = normalize(collab, norm='l2', axis=1)
    n_items = collab_norm.shape[0]
    print(f"加载协同向量: {collab.shape}")

    # 计算余弦相似度矩阵
    print(f"计算 {n_items}x{n_items} 相似度矩阵...")
    sim_mat = collab_norm @ collab_norm.T

    # 排除自身
    np.fill_diagonal(sim_mat, -2.0)

    # 每个 item 取 top-K 邻居
    print(f"提取 top-{K} 邻居...")
    neighbors = {}
    for i in range(n_items):
        topk_idx = np.argpartition(-sim_mat[i], K)[:K]
        # 按相似度排序
        topk_idx = topk_idx[np.argsort(-sim_mat[i, topk_idx])]
        neighbors[i] = topk_idx.tolist()

    # 保存
    out_path = os.path.join(args.data_dir, ds, f'{ds}.collab_neighbors_k{K}.json')
    with open(out_path, 'w') as f:
        json.dump(neighbors, f)
    print(f"已保存: {out_path} ({len(neighbors)} items, K={K})")

    # 简单统计
    all_sims = []
    for i, nbs in neighbors.items():
        for j in nbs:
            all_sims.append(sim_mat[i, j])
    all_sims = np.array(all_sims)
    print(f"邻居间平均 cosine sim: {all_sims.mean():.4f} (min={all_sims.min():.4f}, max={all_sims.max():.4f})")


if __name__ == '__main__':
    main()
