"""
三路 Ensemble: text-code + image-code + SASRec 协同。
SASRec 路直接用 item collab embeddings 为每个用户检索 top-K 候选。
突破 beam=20 的候选池限制。

用法:
    python ensemble_triple.py \
        --output_dir log/Instruments-R1_E1_st-concat-caq \
        --dataset Instruments \
        --data_path ./data/ \
        --index_file .index_lemb_R1_E1_st-concat-caq.json \
        --image_index_file .index_vitemb_R1_E1_st-concat-caq.json \
        --num_beams 20 \
        --collab_topk 20 \
        --collab_weight 0.5
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict

from evaluate import get_metrics_results


def load_collab_and_inters(data_path, dataset):
    emb = np.load(os.path.join(data_path, dataset, f"{dataset}.emb-collab-256.npy"))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_normed = emb / norms
    
    with open(os.path.join(data_path, dataset, f"{dataset}.inter.json"), 'r') as f:
        inters = json.load(f)
    
    return emb_normed, inters


def get_collab_candidates(user_id, inters, collab_emb_normed, topk, 
                          max_his_len=20, strategy="exp_decay"):
    """用 SASRec collab embeddings 为用户检索 topk 候选 (排除历史 items)"""
    items = inters[str(user_id)]
    history = items[:-1]
    if max_his_len > 0:
        history = history[-max_his_len:]
    
    history_set = set(items)  # 排除所有已交互 items (含 target, 但 target 应在结果中)
    # 只排除非 target 的历史 items
    history_set = set(items[:-1])
    
    if len(history) == 0:
        return []
    
    # 指数衰减加权用户表征
    if strategy == "last1":
        user_emb = collab_emb_normed[history[-1]]
    elif strategy == "exp_decay":
        decay = 0.7
        n = len(history)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        user_emb = (collab_emb_normed[history] * weights[:, None]).sum(axis=0)
    else:
        user_emb = collab_emb_normed[history].mean(axis=0)
    
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    
    # 计算与所有 items 的相似度
    sims = collab_emb_normed @ user_emb  # (num_items,)
    
    # 排除历史 items
    for h in history_set:
        sims[h] = -np.inf
    
    # 取 topk
    topk_indices = np.argsort(sims)[-topk:][::-1]
    topk_scores = sims[topk_indices]
    
    return list(zip(topk_indices.tolist(), topk_scores.tolist()))


def triple_ensemble(text_save_file, image_save_file,
                    text_code2item, image_code2item,
                    collab_emb_normed, inters,
                    num_beams, collab_topk, collab_weight,
                    collab_strategy="exp_decay"):
    """三路 ensemble: text + image + collab"""
    text_data = json.load(open(text_save_file, 'r'))
    image_data = json.load(open(image_save_file, 'r'))
    
    t_outputs = text_data['all_outputs']
    t_scores = text_data['all_scores']
    t_targets = text_data['all_targets']
    t_users = text_data['all_users']
    
    i_outputs = image_data['all_outputs']
    i_scores = image_data['all_scores']
    i_users = image_data['all_users']
    
    t_user2idx = {u: idx for idx, u in enumerate(t_users)}
    i_user2idx = {u: idx for idx, u in enumerate(i_users)}
    
    num_users = len(t_users)
    results = []
    collab_hits = 0  # 统计 collab 路独有的 target 命中
    
    for idx in range(num_users):
        user_id = t_users[idx]
        t_i = t_user2idx[user_id]
        i_i = i_user2idx[user_id]
        
        target_code = t_targets[t_i].strip().replace(" ", "")
        target_item_id = text_code2item.get(target_code, -1)
        
        item_scores = {}
        
        # Text path
        for j in range(num_beams):
            code_str = t_outputs[t_i * num_beams + j].strip().replace(" ", "")
            gen_score = t_scores[t_i * num_beams + j]
            item_id = text_code2item.get(code_str, -1)
            if item_id == -1:
                continue
            if item_id in item_scores:
                item_scores[item_id] = (gen_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = gen_score
        
        # Image path
        for j in range(num_beams):
            code_str = i_outputs[i_i * num_beams + j].strip().replace(" ", "")
            gen_score = i_scores[i_i * num_beams + j]
            item_id = image_code2item.get(code_str, -1)
            if item_id == -1:
                continue
            if item_id in item_scores:
                item_scores[item_id] = (gen_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = gen_score
        
        # Collab path (第三路)
        text_image_items = set(item_scores.keys())
        collab_candidates = get_collab_candidates(
            user_id, inters, collab_emb_normed, collab_topk,
            strategy=collab_strategy
        )
        
        for item_id, cos_sim in collab_candidates:
            collab_score = collab_weight * cos_sim
            if item_id in item_scores:
                # 已有两路的分数 → 融合
                item_scores[item_id] = (item_scores[item_id] + collab_score) / 2 + 0.5
            else:
                # 新引入的候选
                item_scores[item_id] = collab_score
                if item_id == target_item_id:
                    collab_hits += 1
        
        # 排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        one_results = []
        for item_id, _ in sorted_items:
            if item_id == target_item_id:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)
    
    return results, collab_hits


def main(args):
    metrics = args.metrics.split(",")
    
    print(f"加载协同嵌入和交互数据: {args.dataset}")
    collab_emb_normed, inters = load_collab_and_inters(args.data_path, args.dataset)
    
    idx_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}{args.index_file}")
    text_code2item = {}
    for item_id, tokens in json.load(open(idx_path)).items():
        text_code2item[''.join(tokens)] = int(item_id)
    
    idx_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}{args.image_index_file}")
    image_code2item = {}
    for item_id, tokens in json.load(open(idx_path)).items():
        image_code2item[''.join(tokens)] = int(item_id)
    
    text_save = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    image_save = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    
    collab_weights = [float(w) for w in args.collab_weight.split(",")]
    collab_topks = [int(k) for k in args.collab_topk.split(",")]
    strategies = args.strategies.split(",")
    
    # 先打印原始 ensemble baseline
    print("\n[原始 2-path ensemble baseline]")
    from ensemble import get_sort_results, get_topk_results_ensemble
    text_data = json.load(open(text_save, 'r'))
    image_data = json.load(open(image_save, 'r'))
    
    t_targets_ids = []
    for t in text_data['all_targets']:
        t_clean = t.strip().replace(" ", "")
        t_targets_ids.append([text_code2item.get(t_clean, -1)])
    i_targets_ids = []
    for t in image_data['all_targets']:
        t_clean = t.strip().replace(" ", "")
        i_targets_ids.append([image_code2item.get(t_clean, -1)])
    
    text_info = get_sort_results(text_data['all_outputs'], text_data['all_scores'],
                                 t_targets_ids, text_data['all_users'], 20, 
                                 {k: [v] for k, v in text_code2item.items()})
    image_info = get_sort_results(image_data['all_outputs'], image_data['all_scores'],
                                  i_targets_ids, image_data['all_users'], 20,
                                  {k: [v] for k, v in image_code2item.items()})
    base_results = get_topk_results_ensemble(text_info, image_info)
    base_m = get_metrics_results(base_results, metrics)
    total = len(base_results)
    for k in base_m:
        base_m[k] /= total
    print(f"  H@1={base_m['hit@1']:.4f}  H@5={base_m['hit@5']:.4f}  H@10={base_m['hit@10']:.4f}  N@5={base_m['ndcg@5']:.4f}  N@10={base_m['ndcg@10']:.4f}")
    
    # 三路 ensemble
    print(f"\n{'strategy':>10s} {'topk':>5s} {'weight':>6s}  {'H@1':>7s}  {'H@5':>7s}  {'H@10':>7s}  {'N@5':>7s}  {'N@10':>7s}  {'collab独有命中':>12s}")
    print("-" * 85)
    
    for strategy in strategies:
        for topk in collab_topks:
            for weight in collab_weights:
                results, collab_hits = triple_ensemble(
                    text_save, image_save,
                    text_code2item, image_code2item,
                    collab_emb_normed, inters,
                    args.num_beams, topk, weight,
                    collab_strategy=strategy
                )
                m = get_metrics_results(results, metrics)
                total = len(results)
                for k in m:
                    m[k] /= total
                
                print(f"{strategy:>10s} {topk:5d} {weight:6.2f}  {m['hit@1']:7.4f}  {m['hit@5']:7.4f}  {m['hit@10']:7.4f}  {m['ndcg@5']:7.4f}  {m['ndcg@10']:7.4f}  {collab_hits:>12d}")


def parse_args():
    parser = argparse.ArgumentParser(description="Triple Ensemble")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--image_index_file", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--collab_topk", type=str, default="10,20,50")
    parser.add_argument("--collab_weight", type=str, default="0.1,0.5,1.0,2.0")
    parser.add_argument("--strategies", type=str, default="exp_decay,last1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
