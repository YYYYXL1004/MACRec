"""
CPA-aware Reranking: 使用训练好的轻量CPA adapter做推理时重排

原理:
  1. 加载带CPA adapter的T5模型
  2. 对每个测试用户，T5 encoder编码输入序列 → mean pool → CPA adapter → 256d user_repr
  3. 对beam中每个候选item，查找SASRec 256d嵌入
  4. 融合: final_score = gen_score + beta * cos_sim(user_repr, item_emb)

用法:
  python rerank_cpa.py \
    --ckpt_path ./log/Instruments-lcpa-cosine-w0.005 \
    --dataset Instruments \
    --collab_emb_path ./data/Instruments/Instruments.emb-collab-256.npy \
    --index_file .index_lemb_R1_E1_st-concat-caq.json \
    --image_index_file .index_vitemb_R1_E1_st-concat-caq.json
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5Config
from modeling import CrossModalContrastive
from evaluate import get_metrics_results, hit_k, ndcg_k


def compute_metrics(predictions, targets):
    """predictions: list of list of code strings, targets: list of code strings"""
    topk_results = []
    for pred_list, target in zip(predictions, targets):
        row = [1 if p == target else 0 for p in pred_list]
        topk_results.append(row)
    n = len(targets)
    return {
        'hit@1': hit_k(topk_results, 1) / n,
        'hit@5': hit_k(topk_results, 5) / n,
        'hit@10': hit_k(topk_results, 10) / n,
        'ndcg@5': ndcg_k(topk_results, 5) / n,
        'ndcg@10': ndcg_k(topk_results, 10) / n,
    }


def build_code_to_itemid(index_path):
    """构建 code_string → item_id 映射"""
    with open(index_path) as f:
        indices = json.load(f)
    code2id = {}
    for item_id_str, tokens in indices.items():
        code_str = ''.join(tokens)
        code2id[code_str] = int(item_id_str)
    return code2id


def load_beam_data(save_path, num_beams=20):
    """加载beam search结果，按用户分组"""
    with open(save_path) as f:
        data = json.load(f)
    
    n_users = len(data['all_targets'])
    users = []
    for i in range(n_users):
        start = i * num_beams
        end = start + num_beams
        users.append({
            'outputs': data['all_outputs'][start:end],
            'scores': data['all_scores'][start:end],
            'target': data['all_targets'][i],
            'user_idx': data['all_users'][i],
        })
    return users


def compute_user_reprs_with_cpa(model, tokenizer, test_data, device, batch_size=128):
    """
    用T5 encoder + CPA adapter 计算所有测试用户的协同表征
    
    返回: (n_users, 256) 的numpy数组
    """
    model.eval()
    all_reprs = []
    
    # 构建输入序列（和测试时一样）
    inputs_list = []
    for d in test_data:
        input_str = ''.join(d["inters"])
        inputs_list.append(input_str)
    
    print(f"  计算 {len(inputs_list)} 个用户的CPA表征...")
    
    with torch.no_grad():
        for i in range(0, len(inputs_list), batch_size):
            batch_texts = inputs_list[i:i+batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                max_length=512,
                truncation=True,
                return_attention_mask=True,
            ).to(device)
            
            # T5 encoder
            encoder_out = model.encoder(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                return_dict=True,
            )
            hidden = encoder_out.last_hidden_state
            
            # mean pool (带 attention mask)
            mask = encoded['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            
            # CPA adapter → 256d
            user_repr = model.cpa_adapter(pooled)
            all_reprs.append(user_repr.cpu().numpy())
            
            if (i // batch_size) % 20 == 0:
                print(f"    batch {i//batch_size}/{len(inputs_list)//batch_size}")
    
    return np.concatenate(all_reprs, axis=0)


def rerank_with_cpa(beam_data, user_reprs, collab_emb, code2id, beta, topk=10):
    """
    CPA reranking: gen_score + beta * cos_sim(user_repr, item_emb)
    """
    predictions = []
    targets = []
    
    for i, user in enumerate(beam_data):
        user_repr = user_reprs[i]
        user_repr_norm = user_repr / (np.linalg.norm(user_repr) + 1e-9)
        
        candidates = []
        for j, (code_str, gen_score) in enumerate(zip(user['outputs'], user['scores'])):
            item_id = code2id.get(code_str, -1)
            if item_id >= 0 and item_id < len(collab_emb):
                item_emb = collab_emb[item_id]
                item_emb_norm = item_emb / (np.linalg.norm(item_emb) + 1e-9)
                cos_sim = np.dot(user_repr_norm, item_emb_norm)
            else:
                cos_sim = 0.0
            
            final_score = gen_score + beta * cos_sim
            candidates.append((code_str, final_score))
        
        # 按融合分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        pred = [c[0] for c in candidates[:topk]]
        predictions.append(pred)
        targets.append(user['target'])
    
    return predictions, targets


def ensemble_rerank(text_beam, image_beam, user_reprs, collab_emb, 
                    text_code2id, image_code2id, beta):
    """
    2-path ensemble + CPA reranking
    按照原ensemble.py逻辑：在item_id空间做融合
    """
    topk_results = []
    
    for i in range(len(text_beam)):
        user_repr = user_reprs[i]
        user_repr_norm = user_repr / (np.linalg.norm(user_repr) + 1e-9)
        
        # 在 item_id 空间合并两路分数（复刻 ensemble.py 逻辑）
        item_score = {}
        
        for code_str, gen_score in zip(text_beam[i]['outputs'], text_beam[i]['scores']):
            item_id = text_code2id.get(code_str, -1)
            if item_id == -1:
                continue
            if item_id in item_score:
                item_score[item_id] = (gen_score + item_score[item_id]) / 2 + 1
            else:
                item_score[item_id] = gen_score
        
        for code_str, gen_score in zip(image_beam[i]['outputs'], image_beam[i]['scores']):
            item_id = image_code2id.get(code_str, -1)
            if item_id == -1:
                continue
            if item_id in item_score:
                item_score[item_id] = (gen_score + item_score[item_id]) / 2 + 1
            else:
                item_score[item_id] = gen_score
        
        # 加入CPA cosine sim
        candidates = []
        for item_id, combined_score in item_score.items():
            if 0 <= item_id < len(collab_emb):
                item_emb = collab_emb[item_id]
                item_emb_norm = item_emb / (np.linalg.norm(item_emb) + 1e-9)
                cos_sim = float(np.dot(user_repr_norm, item_emb_norm))
            else:
                cos_sim = 0.0
            
            final_score = combined_score + beta * cos_sim
            candidates.append((item_id, final_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 目标 item_id
        target_ids = text_code2id.get(text_beam[i]['target'], -1)
        
        # 转为 0/1 列表
        row = []
        for item_id, _ in candidates:
            row.append(1 if item_id == target_ids else 0)
        topk_results.append(row)
    
    return topk_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--collab_emb_path', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--image_index_file', type=str, required=True)
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    data_dir = os.path.join(args.data_path, args.dataset)
    
    # 1. 加载模型（含CPA adapter）
    print("加载模型...")
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    
    collab_emb = np.load(args.collab_emb_path)
    
    model = CrossModalContrastive(config)
    model.resize_token_embeddings(len(tokenizer))
    model.init_lightweight_cpa(collab_emb, cpa_weight=0.0)
    
    # 加载训练好的权重（包含CPA adapter）
    from safetensors.torch import load_file
    ckpt_file = os.path.join(args.ckpt_path, 'model.safetensors')
    if not os.path.exists(ckpt_file):
        ckpt_file = os.path.join(args.ckpt_path, 'pytorch_model.bin')
        state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    else:
        state_dict = load_file(ckpt_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")
    model = model.to(device).eval()
    print(f"  CPA adapter loaded: {model.cpa_adapter.weight.shape}")
    
    # 2. 构建 code → item_id 映射（用text index）
    text_index_path = os.path.join(data_dir, f"{args.dataset}{args.index_file}")
    image_index_path = os.path.join(data_dir, f"{args.dataset}{args.image_index_file}")
    text_code2id = build_code_to_itemid(text_index_path)
    image_code2id = build_code_to_itemid(image_index_path)
    # 合并两套映射
    code2id = {**text_code2id, **image_code2id}
    print(f"  Code→ID映射: text={len(text_code2id)}, image={len(image_code2id)}, total={len(code2id)}")
    
    # 3. 加载beam search结果
    text_save = os.path.join(args.ckpt_path, f'save_seqrec_{args.num_beams}.json')
    image_save = os.path.join(args.ckpt_path, f'save_seqimage_{args.num_beams}.json')
    
    print("加载beam结果...")
    text_beam = load_beam_data(text_save, args.num_beams)
    image_beam = load_beam_data(image_save, args.num_beams)
    print(f"  text beam: {len(text_beam)} users, image beam: {len(image_beam)} users")
    
    # 4. 构建测试数据（用于encoder输入）
    print("构建测试数据...")
    with open(os.path.join(data_dir, f"{args.dataset}.inter.json")) as f:
        inters = json.load(f)
    with open(os.path.join(data_dir, f"{args.dataset}{args.index_file}")) as f:
        indices = json.load(f)
    
    # test set: 每个用户取最后一个item作为target, 前面的作为history
    test_data = []
    for uid, items in inters.items():
        remapped = ["".join(indices[str(i)]) for i in items]
        history = remapped[:-1]
        history = history[-20:]  # max_his_len=20
        test_data.append({"inters": history})
    print(f"  测试用户数: {len(test_data)}")
    
    # 5. 计算CPA用户表征
    print("计算CPA用户表征...")
    user_reprs = compute_user_reprs_with_cpa(model, tokenizer, test_data, device, args.batch_size)
    print(f"  用户表征shape: {user_reprs.shape}")
    print(f"  表征L2 norm均值: {np.linalg.norm(user_reprs, axis=1).mean():.4f}")
    
    # 6. 多组beta做reranking
    print("\n" + "="*60)
    print("CPA Reranking 结果")
    print("="*60)
    
    betas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # --- 单路 text reranking ---
    print("\n--- Text beam (seqrec) + CPA rerank ---")
    for beta in betas:
        preds, tgts = rerank_with_cpa(text_beam, user_reprs, collab_emb, code2id, beta)
        metrics = compute_metrics(preds, tgts)
        print(f"  beta={beta:<5} | H@1={metrics['hit@1']:.4f} H@5={metrics['hit@5']:.4f} "
              f"H@10={metrics['hit@10']:.4f} N@5={metrics['ndcg@5']:.4f} N@10={metrics['ndcg@10']:.4f}")
    
    # --- 2-path ensemble + CPA reranking ---
    print("\n--- 2-path Ensemble + CPA rerank ---")
    n_users = len(text_beam)
    for beta in betas:
        topk_res = ensemble_rerank(text_beam, image_beam, user_reprs, collab_emb, 
                                   text_code2id, image_code2id, beta)
        metrics = {
            'hit@1': hit_k(topk_res, 1) / n_users,
            'hit@5': hit_k(topk_res, 5) / n_users,
            'hit@10': hit_k(topk_res, 10) / n_users,
            'ndcg@5': ndcg_k(topk_res, 5) / n_users,
            'ndcg@10': ndcg_k(topk_res, 10) / n_users,
        }
        tag = " ← baseline" if beta == 0 else ""
        print(f"  beta={beta:<5} | H@1={metrics['hit@1']:.4f} H@5={metrics['hit@5']:.4f} "
              f"H@10={metrics['hit@10']:.4f} N@5={metrics['ndcg@5']:.4f} N@10={metrics['ndcg@10']:.4f}{tag}")
    
    # 保存最佳结果
    print("\n完成!")


if __name__ == '__main__':
    main()
