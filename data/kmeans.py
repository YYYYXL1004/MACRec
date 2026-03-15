"""
KMeans 聚类生成伪标签 (供 CrossRQVAE 对比学习使用)

改进: 拼接 L2 归一化后的协同向量，使伪标签同时反映语义和行为相似度

用法:
    cd MACRec/data
    python kmeans.py --dataset Instruments
"""

import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--n_clusters', type=int, default=512)
    # 文本向量: 使用 sentence-transformer 768 维 (替代 llama 4096 维)
    parser.add_argument('--text_emb_suffix', type=str, default='.emb-st-768.npy')
    parser.add_argument('--image_emb_suffix', type=str, default='.emb-ViT-L-14.npy')
    parser.add_argument('--collab_emb_suffix', type=str, default='.emb-collab-256.npy')
    return parser.parse_args()


def l2_normalize(emb):
    """L2 归一化，零向量保持零"""
    return normalize(emb, norm='l2', axis=1)


def main():
    args = parse_args()
    ds = args.dataset
    n_clusters = args.n_clusters

    # 加载三路向量
    text_emb = np.load(f"{ds}/{ds}{args.text_emb_suffix}")
    image_emb = np.load(f"{ds}/{ds}{args.image_emb_suffix}")
    collab_emb = np.load(f"{ds}/{ds}{args.collab_emb_suffix}")

    print(f"text  shape: {text_emb.shape}, norm 均值: {np.linalg.norm(text_emb, axis=1).mean():.4f}")
    print(f"image shape: {image_emb.shape}, norm 均值: {np.linalg.norm(image_emb, axis=1).mean():.4f}")
    print(f"collab shape: {collab_emb.shape}, norm 均值: {np.linalg.norm(collab_emb, axis=1).mean():.4f}")

    # L2 归一化，消除量纲差异
    text_norm = l2_normalize(text_emb)
    image_norm = l2_normalize(image_emb)
    collab_norm = l2_normalize(collab_emb)

    # 拼接: 语义向量 + 协同向量
    text_concat = np.concatenate([text_norm, collab_norm], axis=1)
    image_concat = np.concatenate([image_norm, collab_norm], axis=1)
    print(f"\n拼接后 text  shape: {text_concat.shape}")
    print(f"拼接后 image shape: {image_concat.shape}")

    # 分别聚类
    for name, emb_concat in [("text", text_concat), ("image", image_concat)]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_concat)

        item2cluster = {}
        for idx, label in enumerate(labels):
            item2cluster[f"{idx}"] = [f"<a_{label}>"]

        if name == "text":
            out_path = f"{ds}/{ds}.index_lemb_kmeans{n_clusters}.json"
        else:
            out_path = f"{ds}/{ds}.index_vitemb_kmeans{n_clusters}.json"

        with open(out_path, "w") as f:
            json.dump(item2cluster, f, indent=2, ensure_ascii=False)
        print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
