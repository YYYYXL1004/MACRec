"""
用 sentence-transformers 生成 768 维文本嵌入 (L2归一化)

输入: data/{Dataset}/{Dataset}.item.json
输出: data/{Dataset}/{Dataset}.emb-st-768.npy  shape=(N_items, 768), 已L2归一化

用法:
    python gen_text_emb_st.py --dataset Instruments
    python gen_text_emb_st.py --dataset Arts --gpu_id 1
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_root = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments',
                        help='Instruments / Arts / Games')
    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--model_name', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='sentence-transformers 模型名，768维输出')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()


def build_item_texts(item_json_path):
    """读取 item.json，拼接 title + description 作为文本，按 item_id 排序"""
    with open(item_json_path, 'r') as f:
        items = json.load(f)

    n_items = len(items)
    texts = [''] * n_items
    for item_id_str, meta in items.items():
        idx = int(item_id_str)
        title = meta.get('title', '').strip()
        desc = meta.get('description', '').strip()
        # title + description，缺失则只用有的部分
        text = f"{title} {desc}".strip() if desc else title
        if not text:
            text = meta.get('brand', '') or meta.get('categories', '') or 'unknown'
        texts[idx] = text

    print(f"共 {n_items} 个 item，文本示例:")
    print(f"  [0] {texts[0][:100]}...")
    print(f"  [1] {texts[1][:100]}...")
    return texts


def main():
    args = parse_args()
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    item_json_path = os.path.join(args.data_root, args.dataset, f'{args.dataset}.item.json')
    output_path = os.path.join(args.data_root, args.dataset, f'{args.dataset}.emb-st-768.npy')

    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model_name}")
    print(f"设备: {device}")

    # 1. 构建文本
    texts = build_item_texts(item_json_path)

    # 2. 加载模型
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_name, device=device)

    # 3. 编码（all-mpnet-base-v2 输出天然 L2 归一化）
    print(f"编码 {len(texts)} 条文本...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 确保 L2 归一化
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # 4. 验证
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"输出 shape: {embeddings.shape}")
    print(f"L2 norm 均值: {norms.mean():.4f} (应接近 1.0)")

    # 5. 保存
    np.save(output_path, embeddings)
    print(f"已保存: {output_path}")


if __name__ == '__main__':
    main()
