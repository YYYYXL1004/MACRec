"""Beam-50 候选池分析 + collab cosine sim reranking"""
import json
import os
import numpy as np
from collections import defaultdict

def load_beam_data(save_file, num_beams):
    data = json.load(open(save_file))
    return data['all_outputs'], data['all_scores'], data['all_targets'], data['all_users']

def analyze_coverage(text_outputs, text_scores, text_targets, 
                     image_outputs, image_scores, image_targets,
                     text_code2id, image_code2id, num_beams):
    """分析目标item在beam候选池中的覆盖率"""
    n = len(text_targets)
    text_hits = 0
    image_hits = 0
    union_hits = 0
    
    for i in range(n):
        text_cands = set()
        for j in range(num_beams):
            code = text_outputs[i*num_beams + j].strip().replace(" ", "")
            if code in text_code2id:
                for item_id in text_code2id[code]:
                    text_cands.add(item_id)
        
        image_cands = set()
        for j in range(num_beams):
            code = image_outputs[i*num_beams + j].strip().replace(" ", "")
            if code in image_code2id:
                for item_id in image_code2id[code]:
                    image_cands.add(item_id)
        
        target_code = text_targets[i].strip().replace(" ", "")
        target_ids = set(text_code2id.get(target_code, []))
        
        if target_ids & text_cands:
            text_hits += 1
        if target_ids & image_cands:
            image_hits += 1
        if target_ids & (text_cands | image_cands):
            union_hits += 1
    
    print(f"总用户数: {n}")
    print(f"目标在 text beam-{num_beams} 中: {text_hits/n*100:.1f}% ({text_hits})")
    print(f"目标在 image beam-{num_beams} 中: {image_hits/n*100:.1f}% ({image_hits})")
    print(f"目标在任一 beam 中 (覆盖率): {union_hits/n*100:.1f}% ({union_hits})")
    return union_hits / n

def rerank_beam50_with_collab(output_dir, dataset, data_path, index_file, image_index_file, 
                               collab_emb_path, num_beams=50, betas=[0, 0.05, 0.1, 0.2, 0.3]):
    """在 beam-50 候选池上做 collab cosine sim reranking"""
    from evaluate import hit_k, ndcg_k
    
    # 加载 index 映射
    index_text = os.path.join(data_path, dataset, f'{dataset}{index_file}')
    index_image = os.path.join(data_path, dataset, f'{dataset}{image_index_file}')
    
    item_id2text_code = json.load(open(index_text))
    text_code2id = defaultdict(list)
    for item_id, codes in item_id2text_code.items():
        text_code2id[''.join(codes)].append(int(item_id))
    
    item_id2image_code = json.load(open(index_image))
    image_code2id = defaultdict(list)
    for item_id, codes in item_id2image_code.items():
        image_code2id[''.join(codes)].append(int(item_id))
    
    # 加载 beam 数据
    text_outputs, text_scores, text_targets, text_users = load_beam_data(
        os.path.join(output_dir, f'save_seqrec_{num_beams}.json'), num_beams)
    image_outputs, image_scores, image_targets, image_users = load_beam_data(
        os.path.join(output_dir, f'save_seqimage_{num_beams}.json'), num_beams)
    
    # 覆盖率分析
    print("=== Beam-50 候选池覆盖率 ===")
    coverage = analyze_coverage(text_outputs, text_scores, text_targets,
                                image_outputs, image_scores, image_targets,
                                text_code2id, image_code2id, num_beams)
    
    # 加载 collab embeddings
    collab_emb = np.load(collab_emb_path)
    collab_emb_norm = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-8)
    
    # 加载交互数据构建用户历史
    inter_file = os.path.join(data_path, dataset, f'{dataset}.inter.json')
    with open(inter_file) as f:
        inters = json.load(f)
    
    n = len(text_targets)
    
    for beta in betas:
        topk_results = []
        
        for i in range(n):
            # 合并 text + image beam 到 item_id 空间
            item_id2score = {}
            
            for j in range(num_beams):
                code = text_outputs[i*num_beams + j].strip().replace(" ", "")
                score = text_scores[i*num_beams + j]
                if code in text_code2id:
                    for item_id in text_code2id[code]:
                        if item_id == -1:
                            continue
                        if item_id in item_id2score:
                            item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
                        else:
                            item_id2score[item_id] = score
                            
            for j in range(num_beams):
                code = image_outputs[i*num_beams + j].strip().replace(" ", "")
                score = image_scores[i*num_beams + j]
                if code in image_code2id:
                    for item_id in image_code2id[code]:
                        if item_id == -1:
                            continue
                        if item_id in item_id2score:
                            item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
                        else:
                            item_id2score[item_id] = score
            
            # Collab reranking
            if beta > 0:
                user = text_users[i]
                user_key = str(user)
                if user_key in inters and len(inters[user_key]) > 0:
                    hist_items = [int(x) for x in inters[user_key][:-1]]
                    hist_items = hist_items[-20:]
                    if hist_items:
                        valid_hist = [h for h in hist_items if h < len(collab_emb_norm)]
                        if valid_hist:
                            user_emb = collab_emb_norm[valid_hist].mean(axis=0)
                            user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
                            
                            for item_id in item_id2score:
                                if 0 <= item_id < len(collab_emb_norm):
                                    cos_sim = np.dot(user_emb, collab_emb_norm[item_id])
                                    item_id2score[item_id] += beta * cos_sim
            
            # 获取 target item id
            target_code = text_targets[i].strip().replace(" ", "")
            target_ids = set(text_code2id.get(target_code, []))
            
            # 排序
            sorted_items = sorted(item_id2score.items(), key=lambda x: x[1], reverse=True)
            
            one_results = []
            for item_id, score in sorted_items:
                one_results.append(1 if item_id in target_ids else 0)
            topk_results.append(one_results)
        
        # 计算指标
        metrics = {
            'hit@1': hit_k(topk_results, 1) / n,
            'hit@5': hit_k(topk_results, 5) / n,
            'hit@10': hit_k(topk_results, 10) / n,
            'ndcg@5': ndcg_k(topk_results, 5) / n,
            'ndcg@10': ndcg_k(topk_results, 10) / n,
        }
        print(f"\nbeta={beta}: {metrics}")

if __name__ == '__main__':
    output_dir = '/data/yaoxianglin/MACRec/log/Instruments-lcpa-cosine-w0.005'
    dataset = 'Instruments'
    data_path = '/data/yaoxianglin/MACRec/data'
    index_file = '.index_lemb_R1_E1_st-concat-caq.json'
    image_index_file = '.index_vitemb_R1_E1_st-concat-caq.json'
    collab_emb_path = '/data/yaoxianglin/MACRec/data/Instruments/Instruments.emb-collab-256.npy'
    
    rerank_beam50_with_collab(
        output_dir, dataset, data_path, 
        index_file, image_index_file, collab_emb_path,
        num_beams=50,
        betas=[0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
