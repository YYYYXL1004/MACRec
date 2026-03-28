"""
加权 Ensemble: 调整 text/image 路的权重比例。
在原 ensemble.py 基础上增加 alpha 参数控制 text 路权重。

用法:
    python ensemble_weighted.py \
        --output_dir log/Instruments-R1_E1_st-concat-caq \
        --dataset Instruments \
        --data_path ./data/ \
        --index_file .index_lemb_R1_E1_st-concat-caq.json \
        --image_index_file .index_vitemb_R1_E1_st-concat-caq.json \
        --num_beams 20 \
        --alphas "0.3,0.4,0.5,0.6,0.7"
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict

from evaluate import get_metrics_results


def weighted_ensemble(text_save_file, image_save_file,
                      text_index2item, image_index2item,
                      num_beams, alpha):
    """
    加权 ensemble。
    alpha: text 路的权重 (image 路为 1-alpha)。
    当两路都命中同一 item 时，用加权平均分 + 1.0 的 bonus。
    """
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
    
    # text targets 用于找 ground truth item_id
    t_target_codes = [t.strip().replace(" ", "") for t in t_targets]
    
    num_users = len(t_users)
    results = []
    
    for idx in range(num_users):
        user_id = t_users[idx]
        t_i = t_user2idx[user_id]
        i_i = i_user2idx[user_id]
        
        target_code = t_target_codes[t_i]
        target_item_id = text_index2item.get(target_code, -1)
        
        item_scores = {}
        item_count = {}
        
        # Text path (权重 alpha)
        for j in range(num_beams):
            code_str = t_outputs[t_i * num_beams + j].strip().replace(" ", "")
            gen_score = t_scores[t_i * num_beams + j]
            
            item_id = text_index2item.get(code_str, -1)
            if item_id == -1:
                continue
            
            weighted_score = alpha * gen_score
            
            if item_id in item_scores:
                # 两路都有 → 加权平均 + bonus
                old_score = item_scores[item_id]
                old_count = item_count[item_id]
                item_scores[item_id] = (old_score * old_count + weighted_score) / (old_count + alpha) + 1.0
                item_count[item_id] = old_count + alpha
            else:
                item_scores[item_id] = weighted_score
                item_count[item_id] = alpha
        
        # Image path (权重 1-alpha)
        img_weight = 1.0 - alpha
        for j in range(num_beams):
            code_str = i_outputs[i_i * num_beams + j].strip().replace(" ", "")
            gen_score = i_scores[i_i * num_beams + j]
            
            item_id = image_index2item.get(code_str, -1)
            if item_id == -1:
                continue
            
            weighted_score = img_weight * gen_score
            
            if item_id in item_scores:
                old_score = item_scores[item_id]
                old_count = item_count[item_id]
                item_scores[item_id] = (old_score * old_count + weighted_score) / (old_count + img_weight) + 1.0
                item_count[item_id] = old_count + img_weight
            else:
                item_scores[item_id] = weighted_score
                item_count[item_id] = img_weight
        
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
    
    # 构建 code -> item_id 映射
    idx_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}{args.index_file}")
    item2code = json.load(open(idx_path, 'r'))
    text_code2item = {}
    for item_id, code_tokens in item2code.items():
        text_code2item[''.join(code_tokens)] = int(item_id)
    
    idx_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}{args.image_index_file}")
    item2code = json.load(open(idx_path, 'r'))
    image_code2item = {}
    for item_id, code_tokens in item2code.items():
        image_code2item[''.join(code_tokens)] = int(item_id)
    
    text_save = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    image_save = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    
    alphas = [float(a) for a in args.alphas.split(",")]
    
    print(f"{'alpha':>6s}  {'H@1':>7s}  {'H@5':>7s}  {'H@10':>7s}  {'N@5':>7s}  {'N@10':>7s}")
    print("-" * 52)
    
    for alpha in alphas:
        results = weighted_ensemble(
            text_save, image_save,
            text_code2item, image_code2item,
            args.num_beams, alpha
        )
        m = get_metrics_results(results, metrics)
        total = len(results)
        for k in m:
            m[k] /= total
        
        print(f"{alpha:6.2f}  {m.get('hit@1',0):7.4f}  {m.get('hit@5',0):7.4f}  {m.get('hit@10',0):7.4f}  {m.get('ndcg@5',0):7.4f}  {m.get('ndcg@10',0):7.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted Ensemble")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--image_index_file", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--alphas", type=str, default="0.3,0.4,0.5,0.6,0.7")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
