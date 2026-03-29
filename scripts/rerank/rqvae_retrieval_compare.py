"""
对比 CAGRec vs MACRec 的 RQVAE 重建空间检索质量。

核心问题：CAGRec 的协同语义空间做检索，是否比 MACRec 的纯内容空间更有效？

检索方式：
  - 用户表征 = 历史 item 的 RQVAE 重建嵌入的指数衰减加权均值
  - 候选检索 = cosine(user_repr, all_items) → top-K
  - 三路 ensemble = text beam + image beam + RQVAE 检索

对比:
  1. CAGRec RQVAE 空间 (1024d, collab+content) 做检索 → 三路 ensemble
  2. MACRec RQVAE 空间 (4096d text / 768d image, 纯内容) 做检索 → 三路 ensemble
  3. 原始 SASRec 嵌入 (256d) 做检索 → 三路 ensemble (之前的结果作为参考)
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict

from evaluate import get_metrics_results


def load_and_normalize(path):
    """加载嵌入并 L2 normalize"""
    emb = np.load(path)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def compute_user_repr(user_id, inters, emb_normed, max_his_len=20):
    """指数衰减加权用户表征"""
    items = inters[str(user_id)]
    history = items[:-1]
    if max_his_len > 0:
        history = history[-max_his_len:]
    if len(history) == 0:
        return np.zeros(emb_normed.shape[1])
    
    decay = 0.7
    n = len(history)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    user_emb = (emb_normed[history] * weights[:, None]).sum(axis=0)
    
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    return user_emb


def batch_compute_user_reprs(uid_list, inters, emb_normed, max_his_len=20):
    """批量计算所有用户的表征，返回 (num_users, dim) 矩阵"""
    dim = emb_normed.shape[1]
    decay = 0.7
    user_embs = np.zeros((len(uid_list), dim), dtype=np.float32)
    
    for i, uid in enumerate(uid_list):
        items = inters[str(uid)]
        history = items[:-1]
        if max_his_len > 0:
            history = history[-max_his_len:]
        if len(history) == 0:
            continue
        n = len(history)
        weights = np.array([decay ** (n - 1 - j) for j in range(n)])
        weights = weights / weights.sum()
        user_embs[i] = (emb_normed[history] * weights[:, None]).sum(axis=0)
    
    norms = np.linalg.norm(user_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    user_embs = user_embs / norms
    return user_embs


def retrieval_recall(inters, emb_normed, topk_list=[10, 20, 50, 100]):
    """纯检索的 recall: 批量矩阵运算版本"""
    uid_list = list(inters.keys())
    num_users = len(uid_list)
    max_k = max(topk_list)
    
    user_embs = batch_compute_user_reprs([int(u) for u in uid_list], inters, emb_normed)
    
    # 批量计算相似度: (num_users, num_items)
    batch_size = 1000
    hits = {k: 0 for k in topk_list}
    
    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        batch_uids = uid_list[start:end]
        batch_user_embs = user_embs[start:end]
        
        # (batch, dim) @ (dim, items) -> (batch, items)
        sims = batch_user_embs @ emb_normed.T
        
        for i, uid in enumerate(batch_uids):
            items = inters[uid]
            target = items[-1]
            for h in items[:-1]:
                sims[i, h] = -np.inf
            
            topk_items = np.argpartition(sims[i], -max_k)[-max_k:]
            topk_sims = sims[i, topk_items]
            
            for k in topk_list:
                if k >= max_k:
                    if target in topk_items:
                        hits[k] += 1
                else:
                    threshold = np.partition(topk_sims, -k)[-k]
                    if sims[i, target] >= threshold:
                        hits[k] += 1
    
    print(f"  纯检索 Recall:", flush=True)
    for k in topk_list:
        print(f"    Recall@{k}: {hits[k]}/{num_users} = {hits[k]/num_users:.4f}", flush=True)
    return hits


def preload_beam_data(text_save_file, image_save_file,
                      text_code2item, image_code2item, num_beams):
    """预加载 beam 数据，避免每次三路 ensemble 重复加载 JSON"""
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
    
    # 预处理每个用户的 beam 候选
    user_beam_data = {}
    for idx in range(len(t_users)):
        user_id = t_users[idx]
        t_i = t_user2idx[user_id]
        i_i = i_user2idx[user_id]
        
        target_code = t_targets[t_i].strip().replace(" ", "")
        target_item_id = text_code2item.get(target_code, -1)
        
        # Text beam
        text_items = []
        for j in range(num_beams):
            code_str = t_outputs[t_i * num_beams + j].strip().replace(" ", "")
            gen_score = t_scores[t_i * num_beams + j]
            item_id = text_code2item.get(code_str, -1)
            if item_id != -1:
                text_items.append((item_id, gen_score))
        
        # Image beam
        image_items = []
        for j in range(num_beams):
            code_str = i_outputs[i_i * num_beams + j].strip().replace(" ", "")
            gen_score = i_scores[i_i * num_beams + j]
            item_id = image_code2item.get(code_str, -1)
            if item_id != -1:
                image_items.append((item_id, gen_score))
        
        user_beam_data[user_id] = {
            'target': target_item_id,
            'text': text_items,
            'image': image_items,
        }
    
    return t_users, user_beam_data


def triple_ensemble(t_users, user_beam_data,
                    retrieval_emb_normed, inters,
                    user_embs_normed, uid_to_uidx,
                    collab_topk, collab_weight):
    """三路 ensemble: text beam + image beam + RQVAE空间检索（优化版）"""
    num_users = len(t_users)
    results = []
    retrieval_unique_hits = 0
    
    for idx in range(num_users):
        user_id = t_users[idx]
        bd = user_beam_data[user_id]
        target_item_id = bd['target']
        
        item_scores = {}
        
        # Text beam path
        for item_id, gen_score in bd['text']:
            if item_id in item_scores:
                item_scores[item_id] = (gen_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = gen_score
        
        # Image beam path
        for item_id, gen_score in bd['image']:
            if item_id in item_scores:
                item_scores[item_id] = (gen_score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = gen_score
        
        # RQVAE 空间检索 path
        items = inters[str(user_id)]
        history_set = set(items[:-1])
        uidx = uid_to_uidx[user_id]
        user_emb = user_embs_normed[uidx]
        sims = retrieval_emb_normed @ user_emb
        for h in history_set:
            sims[h] = -np.inf
        
        topk_indices = np.argpartition(sims, -collab_topk)[-collab_topk:]
        topk_sims = sims[topk_indices]
        
        for item_id, cos_sim in zip(topk_indices.tolist(), topk_sims.tolist()):
            r_score = collab_weight * cos_sim
            if item_id in item_scores:
                item_scores[item_id] = (item_scores[item_id] + r_score) / 2 + 0.5
            else:
                item_scores[item_id] = r_score
                if item_id == target_item_id:
                    retrieval_unique_hits += 1
        
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        one_results = []
        for item_id, _ in sorted_items:
            if item_id == target_item_id:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)
    
    return results, retrieval_unique_hits


def build_code2item(data_path, dataset, index_file):
    idx_path = os.path.join(data_path, dataset, f"{dataset}{index_file}")
    item2code = json.load(open(idx_path, 'r'))
    code2item = {}
    for item_id, tokens in item2code.items():
        code2item[''.join(tokens)] = int(item_id)
    return code2item


def run_experiment(name, retrieval_emb_path, 
                   t_users, user_beam_data,
                   inters, topks, weights, metrics):
    """运行一组实验: 纯检索 recall + 三路 ensemble"""
    print(f"\n{'#'*70}", flush=True)
    print(f"# {name}", flush=True)
    print(f"# 嵌入: {retrieval_emb_path}", flush=True)
    print(f"{'#'*70}", flush=True)
    
    emb_normed = load_and_normalize(retrieval_emb_path)
    print(f"  嵌入维度: {emb_normed.shape}", flush=True)
    
    # 纯检索 recall (衡量空间质量)
    retrieval_recall(inters, emb_normed, topk_list=[10, 20, 50, 100])
    
    # 批量计算用户表征（所有 topk/weight 共享）
    uid_list = [int(u) for u in t_users]
    uid_to_uidx = {uid: i for i, uid in enumerate(uid_list)}
    print(f"  预计算用户表征...", flush=True)
    user_embs = batch_compute_user_reprs(uid_list, inters, emb_normed)
    
    # 三路 ensemble
    print(f"\n  三路 Ensemble (text beam + image beam + RQVAE检索):", flush=True)
    print(f"  {'topk':>5s} {'weight':>6s}  {'H@1':>7s}  {'H@5':>7s}  {'H@10':>7s}  {'N@5':>7s}  {'N@10':>7s}  {'检索独有命中':>10s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    
    best_n10 = 0
    best_config = ""
    
    for topk in topks:
        for w in weights:
            results, unique_hits = triple_ensemble(
                t_users, user_beam_data,
                emb_normed, inters,
                user_embs, uid_to_uidx,
                topk, w
            )
            m = get_metrics_results(results, metrics)
            total = len(results)
            for k in m:
                m[k] /= total
            
            n10 = m['ndcg@10']
            marker = " ★" if n10 > best_n10 else ""
            if n10 > best_n10:
                best_n10 = n10
                best_config = f"topk={topk}, w={w}"
            
            print(f"  {topk:5d} {w:6.2f}  {m['hit@1']:7.4f}  {m['hit@5']:7.4f}  {m['hit@10']:7.4f}  {m['ndcg@5']:7.4f}  {m['ndcg@10']:7.4f}  {unique_hits:>10d}{marker}", flush=True)
    
    print(f"\n  最佳: {best_config} → NDCG@10={best_n10:.4f}", flush=True)
    return best_n10


def main(args):
    metrics = args.metrics.split(",")
    
    print("加载交互数据...")
    with open(os.path.join(args.data_path, args.dataset, f"{args.dataset}.inter.json")) as f:
        inters = json.load(f)
    print(f"  用户数: {len(inters)}")
    
    topks = [int(k) for k in args.collab_topk.split(",")]
    weights = [float(w) for w in args.collab_weight.split(",")]
    
    # ============================================================
    # 实验 1: CAGRec RQVAE 空间 (协同语义空间) + CAGRec beam
    # ============================================================
    cag_text_code2item = build_code2item(args.data_path, args.dataset, args.cag_index_file)
    cag_image_code2item = build_code2item(args.data_path, args.dataset, args.cag_image_index_file)
    cag_text_save = os.path.join(args.cag_output_dir, f'save_seqrec_{args.num_beams}.json')
    cag_image_save = os.path.join(args.cag_output_dir, f'save_seqimage_{args.num_beams}.json')
    
    # 打印 CAGRec 2-path baseline
    print("\n[CAGRec 2-path ensemble baseline]")
    from ensemble import get_sort_results, get_topk_results_ensemble
    td = json.load(open(cag_text_save)); id_ = json.load(open(cag_image_save))
    t_tgt = [cag_text_code2item.get(t.strip().replace(" ",""), -1) for t in td['all_targets']]
    i_tgt = [cag_image_code2item.get(t.strip().replace(" ",""), -1) for t in id_['all_targets']]
    ti = get_sort_results(td['all_outputs'], td['all_scores'], [[t] for t in t_tgt], td['all_users'], 20, {k:[v] for k,v in cag_text_code2item.items()})
    ii = get_sort_results(id_['all_outputs'], id_['all_scores'], [[t] for t in i_tgt], id_['all_users'], 20, {k:[v] for k,v in cag_image_code2item.items()})
    br = get_topk_results_ensemble(ti, ii)
    bm = get_metrics_results(br, metrics)
    for k in bm: bm[k] /= len(br)
    print(f"  H@1={bm['hit@1']:.4f}  H@5={bm['hit@5']:.4f}  H@10={bm['hit@10']:.4f}  N@5={bm['ndcg@5']:.4f}  N@10={bm['ndcg@10']:.4f}")
    
    # 预加载 CAGRec beam 数据
    print("预加载 CAGRec beam 数据...", flush=True)
    cag_t_users, cag_beam_data = preload_beam_data(
        cag_text_save, cag_image_save,
        cag_text_code2item, cag_image_code2item, args.num_beams)
    
    # CAGRec text-aligned (协同语义空间)
    cag_best = run_experiment(
        "CAGRec RQVAE text-aligned (协同语义空间, 1024d)",
        args.cag_text_emb,
        cag_t_users, cag_beam_data,
        inters, topks, weights, metrics
    )
    
    # CAGRec image-aligned
    run_experiment(
        "CAGRec RQVAE image-aligned (协同语义空间, 1024d)",
        args.cag_image_emb,
        cag_t_users, cag_beam_data,
        inters, topks, weights, metrics
    )
    
    # ============================================================
    # 实验 2: MACRec baseline RQVAE 空间 (纯内容) + MACRec beam
    # ============================================================
    mac_text_code2item = build_code2item(args.data_path, args.dataset, args.mac_index_file)
    mac_image_code2item = build_code2item(args.data_path, args.dataset, args.mac_image_index_file)
    mac_text_save = os.path.join(args.mac_output_dir, f'save_seqrec_{args.num_beams}.json')
    mac_image_save = os.path.join(args.mac_output_dir, f'save_seqimage_{args.num_beams}.json')
    
    # MACRec 2-path baseline
    print("\n[MACRec 2-path ensemble baseline]", flush=True)
    td2 = json.load(open(mac_text_save)); id2 = json.load(open(mac_image_save))
    t_tgt2 = [mac_text_code2item.get(t.strip().replace(" ",""), -1) for t in td2['all_targets']]
    i_tgt2 = [mac_image_code2item.get(t.strip().replace(" ",""), -1) for t in id2['all_targets']]
    ti2 = get_sort_results(td2['all_outputs'], td2['all_scores'], [[t] for t in t_tgt2], td2['all_users'], 20, {k:[v] for k,v in mac_text_code2item.items()})
    ii2 = get_sort_results(id2['all_outputs'], id2['all_scores'], [[t] for t in i_tgt2], id2['all_users'], 20, {k:[v] for k,v in mac_image_code2item.items()})
    br2 = get_topk_results_ensemble(ti2, ii2)
    bm2 = get_metrics_results(br2, metrics)
    for k in bm2: bm2[k] /= len(br2)
    print(f"  H@1={bm2['hit@1']:.4f}  H@5={bm2['hit@5']:.4f}  H@10={bm2['hit@10']:.4f}  N@5={bm2['ndcg@5']:.4f}  N@10={bm2['ndcg@10']:.4f}", flush=True)
    
    # 预加载 MACRec beam 数据
    print("预加载 MACRec beam 数据...", flush=True)
    mac_t_users, mac_beam_data = preload_beam_data(
        mac_text_save, mac_image_save,
        mac_text_code2item, mac_image_code2item, args.num_beams)
    
    # 查找 MACRec baseline 的 aligned 嵌入文件
    data_dir = os.path.join(args.data_path, args.dataset)
    mac_text_aligned = None
    mac_image_aligned = None
    for dim in [4096, 768]:
        p = os.path.join(data_dir, f"{args.dataset}_baseline.emb-text-aligned-{dim}.npy")
        if os.path.exists(p):
            mac_text_aligned = p
            break
    for dim in [768, 4096]:
        p = os.path.join(data_dir, f"{args.dataset}_baseline.emb-image-aligned-{dim}.npy")
        if os.path.exists(p):
            mac_image_aligned = p
            break
    
    if mac_text_aligned:
        run_experiment(
            f"MACRec RQVAE text-aligned (纯内容空间)",
            mac_text_aligned,
            mac_t_users, mac_beam_data,
            inters, topks, weights, metrics
        )
    else:
        print("\n[WARNING] MACRec text-aligned 嵌入未找到，跳过", flush=True)
    
    if mac_image_aligned:
        run_experiment(
            f"MACRec RQVAE image-aligned (纯内容空间)",
            mac_image_aligned,
            mac_t_users, mac_beam_data,
            inters, topks, weights, metrics
        )
    else:
        print("\n[WARNING] MACRec image-aligned 嵌入未找到，跳过", flush=True)
    
    # ============================================================
    # 实验 3: 原始 SASRec 嵌入 (参考基准)
    # ============================================================
    sasrec_path = os.path.join(data_dir, f"{args.dataset}.emb-collab-256.npy")
    if os.path.exists(sasrec_path):
        # 在 CAGRec beam 上
        run_experiment(
            "SASRec 原始嵌入 (256d) + CAGRec beam (之前的三路结果)",
            sasrec_path,
            cag_t_users, cag_beam_data,
            inters, topks, weights, metrics
        )
        # 在 MACRec beam 上
        run_experiment(
            "SASRec 原始嵌入 (256d) + MACRec beam",
            sasrec_path,
            mac_t_users, mac_beam_data,
            inters, topks, weights, metrics
        )
    
    print("\n" + "="*70)
    print("实验完成！对比要点:")
    print("  1. CAGRec RQVAE空间 vs MACRec RQVAE空间: 纯检索 Recall 差异")
    print("  2. CAGRec RQVAE空间 vs SASRec: 检索来源是否重要")
    print("  3. 三路ensemble: 哪个空间检索带来最大提升")
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--cag_text_emb", type=str, required=True)
    parser.add_argument("--cag_image_emb", type=str, required=True)
    parser.add_argument("--cag_output_dir", type=str, required=True)
    parser.add_argument("--cag_index_file", type=str, required=True)
    parser.add_argument("--cag_image_index_file", type=str, required=True)
    parser.add_argument("--mac_output_dir", type=str, required=True)
    parser.add_argument("--mac_index_file", type=str, required=True)
    parser.add_argument("--mac_image_index_file", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--collab_topk", type=str, default="10,20,50")
    parser.add_argument("--collab_weight", type=str, default="0.1,0.3,0.5")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
