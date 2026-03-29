"""
SASRec 协同 reranking: 直接用 SASRec item embeddings 对 beam candidates 做重排。
不依赖 CPA adapter，纯协同信号。

用法:
    python rerank_collab.py \
        --output_dir log/Instruments-R1_E1_st-concat-caq \
        --dataset Instruments \
        --data_path ./data/ \
        --index_file .index_lemb_R1_E1_st-concat-caq.json \
        --image_index_file .index_vitemb_R1_E1_st-concat-caq.json \
        --num_beams 20 \
        --beta 0.1
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict

from evaluate import get_metrics_results


def load_collab_embeddings(data_path, dataset):
    """加载 SASRec 预训练的 item collaborative embeddings"""
    emb_path = os.path.join(data_path, dataset, f"{dataset}.emb-collab-256.npy")
    emb = np.load(emb_path)
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_normed = emb / norms
    return emb, emb_normed


def load_interactions(data_path, dataset):
    """加载用户交互序列"""
    inter_path = os.path.join(data_path, dataset, f"{dataset}.inter.json")
    with open(inter_path, 'r') as f:
        inters = json.load(f)
    return inters


def build_code_to_item(data_path, dataset, index_file):
    """从 index file 构建 code_string -> item_id 映射"""
    idx_path = os.path.join(data_path, dataset, f"{dataset}{index_file}")
    with open(idx_path, 'r') as f:
        item2code = json.load(f)
    code2item = {}
    for item_id, code_tokens in item2code.items():
        code_str = ''.join(code_tokens)
        code2item[code_str] = int(item_id)
    return code2item


def compute_user_repr(user_id, inters, collab_emb_normed, max_his_len=20, strategy="mean"):
    """
    用用户历史 item 的 collab embedding 作为用户表征。
    strategy:
        "mean"     - 全历史均值
        "last1"    - 只用最后1个item
        "last3"    - 最后3个item均值
        "exp_decay"- 指数衰减加权 (越近权重越大, decay=0.7)
    """
    items = inters[str(user_id)]
    # 去掉最后一个 (test target)
    history = items[:-1]
    if max_his_len > 0:
        history = history[-max_his_len:]
    if len(history) == 0:
        return np.zeros(collab_emb_normed.shape[1])
    
    if strategy == "last1":
        user_emb = collab_emb_normed[history[-1]]
    elif strategy == "last3":
        last_n = history[-3:] if len(history) >= 3 else history
        user_emb = collab_emb_normed[last_n].mean(axis=0)
    elif strategy == "exp_decay":
        decay = 0.7
        n = len(history)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        user_emb = (collab_emb_normed[history] * weights[:, None]).sum(axis=0)
    else:  # mean
        user_emb = collab_emb_normed[history].mean(axis=0)
    
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    return user_emb


def rerank_single_path(save_file, code2item, collab_emb_normed, 
                        inters, num_beams, beta, max_his_len=20,
                        strategy="mean", combine="add"):
    """对单路 (text 或 image) 的 beam candidates 做协同 reranking"""
    with open(save_file, 'r') as f:
        save_data = json.load(f)
    
    all_outputs = save_data['all_outputs']
    all_scores = save_data['all_scores']
    all_targets = save_data['all_targets']
    all_users = save_data['all_users']
    
    num_users = len(all_users)
    
    reranked_results = []
    
    for i in range(num_users):
        user_id = all_users[i]
        target = all_targets[i].strip().replace(" ", "")
        
        beam_outputs = all_outputs[i * num_beams: (i + 1) * num_beams]
        beam_scores = all_scores[i * num_beams: (i + 1) * num_beams]
        
        user_emb = compute_user_repr(user_id, inters, collab_emb_normed, max_his_len, strategy)
        
        scored_candidates = []
        for j in range(len(beam_outputs)):
            code_str = beam_outputs[j].strip().replace(" ", "")
            gen_score = beam_scores[j]
            
            item_id = code2item.get(code_str, -1)
            if item_id >= 0:
                item_emb = collab_emb_normed[item_id]
                cos_sim = float(np.dot(user_emb, item_emb))
            else:
                cos_sim = 0.0
            
            if combine == "mul":
                final_score = gen_score * (1.0 + beta * cos_sim)
            else:
                final_score = gen_score + beta * cos_sim
            scored_candidates.append((code_str, final_score))
        
        # 按 final_score 降序排
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        one_results = []
        for code_str, _ in scored_candidates:
            if code_str == target:
                one_results.append(1)
            else:
                one_results.append(0)
        
        reranked_results.append(one_results)
    
    return reranked_results, all_users, all_targets


def rerank_ensemble(text_save_file, image_save_file, 
                    text_code2item, image_code2item,
                    collab_emb_normed, inters,
                    num_beams, beta, max_his_len=20,
                    strategy="mean", combine="add"):
    """对 text+image 双路做协同 reranking 后 ensemble"""
    
    text_data = json.load(open(text_save_file, 'r'))
    image_data = json.load(open(image_save_file, 'r'))
    
    t_outputs = text_data['all_outputs']
    t_scores = text_data['all_scores']
    t_targets = text_data['all_targets']
    t_users = text_data['all_users']
    
    i_outputs = image_data['all_outputs']
    i_scores = image_data['all_scores']
    i_users = image_data['all_users']
    
    # 构建 user -> index 映射 (两路可能顺序不同)
    t_user2idx = {u: idx for idx, u in enumerate(t_users)}
    i_user2idx = {u: idx for idx, u in enumerate(i_users)}
    
    all_users_set = set(t_users) & set(i_users)
    num_users = len(t_users)
    
    results = []
    
    for idx in range(num_users):
        user_id = t_users[idx]
        t_i = t_user2idx[user_id]
        i_i = i_user2idx[user_id]
        
        target_code = t_targets[t_i].strip().replace(" ", "")
        
        user_emb = compute_user_repr(user_id, inters, collab_emb_normed, max_his_len, strategy)
        
        # 收集所有 candidate item_id -> scores
        item_scores = {}
        
        # Text path
        for j in range(num_beams):
            code_str = t_outputs[t_i * num_beams + j].strip().replace(" ", "")
            gen_score = t_scores[t_i * num_beams + j]
            
            item_id = text_code2item.get(code_str, -1)
            if item_id < 0:
                continue
            
            cos_sim = float(np.dot(user_emb, collab_emb_normed[item_id]))
            if combine == "mul":
                final_score = gen_score * (1.0 + beta * cos_sim)
            else:
                final_score = gen_score + beta * cos_sim
            
            if item_id in item_scores:
                item_scores[item_id] = (final_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = final_score
        
        # Image path
        for j in range(num_beams):
            code_str = i_outputs[i_i * num_beams + j].strip().replace(" ", "")
            gen_score = i_scores[i_i * num_beams + j]
            
            item_id = image_code2item.get(code_str, -1)
            if item_id < 0:
                continue
            
            cos_sim = float(np.dot(user_emb, collab_emb_normed[item_id]))
            if combine == "mul":
                final_score = gen_score * (1.0 + beta * cos_sim)
            else:
                final_score = gen_score + beta * cos_sim
            
            if item_id in item_scores:
                item_scores[item_id] = (final_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = final_score
        
        # 找 target item_id
        target_item_id = text_code2item.get(target_code, -1)
        
        # 排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        one_results = []
        for item_id, _ in sorted_items:
            if item_id == target_item_id:
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    
    return results


def main(args):
    metrics = args.metrics.split(",")
    
    print(f"加载协同嵌入: {args.dataset}")
    collab_emb, collab_emb_normed = load_collab_embeddings(args.data_path, args.dataset)
    print(f"  shape: {collab_emb.shape}")
    
    print("加载交互序列")
    inters = load_interactions(args.data_path, args.dataset)
    print(f"  用户数: {len(inters)}")
    
    print("构建 code->item 映射")
    text_code2item = build_code_to_item(args.data_path, args.dataset, args.index_file)
    image_code2item = build_code_to_item(args.data_path, args.dataset, args.image_index_file)
    print(f"  text codes: {len(text_code2item)}, image codes: {len(image_code2item)}")
    
    text_save = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    image_save = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    
    betas = [float(b) for b in args.beta.split(",")]
    strategies = args.strategies.split(",")
    combines = args.combines.split(",")
    
    for strategy in strategies:
        for combine in combines:
            for beta in betas:
                tag = f"s={strategy} c={combine} β={beta}"
                print(f"\n{'='*60}")
                print(tag)
                print(f"{'='*60}")
                
                # --- ensemble + collab rerank (只看最终ensemble) ---
                ensemble_results = rerank_ensemble(
                    text_save, image_save,
                    text_code2item, image_code2item,
                    collab_emb_normed, inters,
                    args.num_beams, beta,
                    strategy=strategy, combine=combine
                )
                ensemble_metrics = get_metrics_results(ensemble_results, metrics)
                total = len(ensemble_results)
                for m in ensemble_metrics:
                    ensemble_metrics[m] /= total
                
                # 紧凑输出
                h1 = ensemble_metrics.get('hit@1', 0)
                h5 = ensemble_metrics.get('hit@5', 0)
                h10 = ensemble_metrics.get('hit@10', 0)
                n5 = ensemble_metrics.get('ndcg@5', 0)
                n10 = ensemble_metrics.get('ndcg@10', 0)
                print(f"  H@1={h1:.4f}  H@5={h5:.4f}  H@10={h10:.4f}  N@5={n5:.4f}  N@10={n10:.4f}")
                
                # 保存
                save_name = f'results_collab_{strategy}_{combine}_beta{beta}.json'
                save_path = os.path.join(args.output_dir, save_name)
                json.dump(ensemble_metrics, open(save_path, 'w'), indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec Collaborative Reranking")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--image_index_file", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--num_beams", type=int, default=20)
    # 支持逗号分隔多个值一次性测试
    parser.add_argument("--beta", type=str, default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--strategies", type=str, default="mean,last1,last3,exp_decay")
    parser.add_argument("--combines", type=str, default="add")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
