"""
高级重排实验：对比多种重排方法在 beam 候选上的效果。

方法:
  1. Baseline: 纯生成分数 (2-path ensemble)
  2. SASRec 模型打分: 用完整 SASRec 模型 P(item|history) 做重排 (非 cosine sim)
  3. RRF (Reciprocal Rank Fusion): 基于排名的融合，不依赖分数量纲
  4. Score Normalization: min-max 归一化后加权融合
  5. Boost-only: 只提升高协同分数的 item，不降低任何 item
  6. Conditional: 仅在生成模型不确定时（top 分数差距小）才介入重排
"""

import json
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# PyTorch 兼容
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from evaluate import get_metrics_results


# ============================================================
# SASRec 模型打分器
# ============================================================
class SASRecScorer:
    """使用 RecBole 原生 SASRec 模型对 (user_history, candidate_items) 打分"""
    
    def __init__(self, ckpt_path, device='cuda'):
        from recbole.model.sequential_recommender import SASRec
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.cfg = ckpt['config']
        self.max_seq_length = self.cfg['max_seq_length']
        self.n_items = ckpt['state_dict']['item_embedding.weight'].shape[0]
        
        class MockDataset:
            def __init__(self, n_items):
                self._n = n_items
            def num(self, field):
                return self._n if field == 'item_id' else 0
        
        self.model = SASRec(self.cfg, MockDataset(self.n_items))
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        self.device = device
        self.model = self.model.to(device)
        
        dim = self.cfg['hidden_size']
        n_layers = self.cfg['n_layers']
        print(f"  SASRec 加载完成: {self.n_items} items, dim={dim}, {n_layers} layers, max_seq={self.max_seq_length}")
    
    @torch.no_grad()
    def batch_score_all_items(self, all_histories, batch_size=512):
        """批量计算所有用户对所有 item 的 SASRec 分数"""
        n_users = len(all_histories)
        # SASRec item_id: 0=padding, 1~N=actual items
        # 我们的 item_id: 0~N-1，需要 +1
        all_item_emb = self.model.item_embedding.weight  # (n_items, dim)
        
        all_scores = np.zeros((n_users, self.n_items), dtype=np.float32)
        
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_histories = all_histories[start:end]
            bs = len(batch_histories)
            
            item_seq = torch.zeros(bs, self.max_seq_length, dtype=torch.long, device=self.device)
            item_seq_len = torch.zeros(bs, dtype=torch.long, device=self.device)
            
            for i, hist in enumerate(batch_histories):
                h = [x + 1 for x in hist[-self.max_seq_length:]]
                sl = len(h)
                item_seq[i, self.max_seq_length - sl:] = torch.tensor(h, dtype=torch.long)
                item_seq_len[i] = sl
            
            # RecBole SASRec.forward 返回最后位置的隐状态
            seq_output = self.model.forward(item_seq, item_seq_len)  # (bs, dim)
            scores = (seq_output @ all_item_emb.T).cpu().numpy()  # (bs, n_items)
            all_scores[start:end] = scores
            
            if (start // batch_size) % 5 == 0:
                print(f"    SASRec 打分进度: {end}/{n_users}", flush=True)
        
        return all_scores


# ============================================================
# 重排方法
# ============================================================

def method_baseline(gen_scores, **kwargs):
    """纯生成分数"""
    return gen_scores.copy()


def method_sasrec_score(gen_scores, sasrec_scores=None, alpha=0.5, **kwargs):
    """SASRec 模型打分融合: final = (1-alpha)*norm(gen) + alpha*norm(sasrec)"""
    # Min-max normalize 各自分数
    g = gen_scores.copy()
    s = sasrec_scores.copy()
    
    g_min, g_max = g.min(), g.max()
    if g_max > g_min:
        g = (g - g_min) / (g_max - g_min)
    
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min)
    
    return (1 - alpha) * g + alpha * s


def method_rrf(gen_scores, sasrec_scores=None, k=60, **kwargs):
    """Reciprocal Rank Fusion: score = 1/(k+rank_gen) + 1/(k+rank_sasrec)"""
    n = len(gen_scores)
    
    gen_ranks = np.zeros(n)
    gen_order = np.argsort(-gen_scores)
    for rank, idx in enumerate(gen_order):
        gen_ranks[idx] = rank + 1
    
    sasrec_ranks = np.zeros(n)
    sasrec_order = np.argsort(-sasrec_scores)
    for rank, idx in enumerate(sasrec_order):
        sasrec_ranks[idx] = rank + 1
    
    return 1.0 / (k + gen_ranks) + 1.0 / (k + sasrec_ranks)


def method_boost_only(gen_scores, sasrec_scores=None, threshold_pct=70, boost=0.1, **kwargs):
    """
    只提升高 SASRec 分数的 item，不降低任何 item 的分数。
    threshold_pct: SASRec 分数超过百分位数才加分
    """
    result = gen_scores.copy()
    threshold = np.percentile(sasrec_scores, threshold_pct)
    mask = sasrec_scores >= threshold
    
    # 将 SASRec 分数归一化到 [0, boost]
    s = sasrec_scores.copy()
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min) * boost
    
    result[mask] += s[mask]
    return result


def method_conditional(gen_scores, sasrec_scores=None, uncertainty_threshold=0.05, alpha=0.3, **kwargs):
    """
    条件重排: 当生成模型不确定时（top-1 和 top-2 分数接近），才用 SASRec 介入。
    否则保持原排序。
    """
    sorted_gen = np.sort(gen_scores)[::-1]
    gap = sorted_gen[0] - sorted_gen[1] if len(sorted_gen) > 1 else float('inf')
    
    if gap < uncertainty_threshold:
        return method_sasrec_score(gen_scores, sasrec_scores=sasrec_scores, alpha=alpha)
    else:
        return gen_scores.copy()


def method_softmax_fusion(gen_scores, sasrec_scores=None, gen_temp=1.0, sas_temp=1.0, alpha=0.5, **kwargs):
    """
    Softmax 归一化后融合: 将两种分数都转为概率分布，然后加权平均。
    这是分布级别的融合，比 min-max 更合理。
    """
    gen_probs = np.exp(gen_scores / gen_temp)
    gen_probs = gen_probs / gen_probs.sum()
    
    sas_probs = np.exp(sasrec_scores / sas_temp)
    sas_probs = sas_probs / sas_probs.sum()
    
    return (1 - alpha) * gen_probs + alpha * sas_probs


# ============================================================
# 主流程
# ============================================================

def run_rerank_experiment(beam_data, inters, sasrec_scorer, metrics, device='cuda'):
    """对 2-path ensemble 的候选做各种重排"""
    
    t_users = beam_data['t_users']
    num_users = len(t_users)
    
    # 预计算所有用户的 SASRec 分数
    print("\n预计算 SASRec 模型分数 (完整前向推理)...", flush=True)
    all_histories = []
    for uid in t_users:
        items = inters[str(uid)]
        all_histories.append(items[:-1])
    
    sasrec_all_scores = sasrec_scorer.batch_score_all_items(all_histories, batch_size=512)
    print(f"  完成，形状: {sasrec_all_scores.shape}", flush=True)
    
    # 同时加载 cosine sim 用的静态嵌入做对比
    collab_emb = np.load(f"data/Instruments/Instruments.emb-collab-256.npy")
    norms = np.linalg.norm(collab_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    collab_emb_normed = collab_emb / norms
    
    # 先分析 SASRec 模型打分 vs cosine sim vs 生成分数的区分能力
    print("\n=== 信号质量分析 ===", flush=True)
    analyze_signal_quality(beam_data, inters, sasrec_all_scores, collab_emb_normed)
    
    # 定义所有重排方法和参数
    methods = [
        ("Baseline (2-path ensemble)", method_baseline, {}),
        # SASRec 模型打分 + 归一化融合
        ("SASRec-score α=0.1", method_sasrec_score, {"alpha": 0.1}),
        ("SASRec-score α=0.2", method_sasrec_score, {"alpha": 0.2}),
        ("SASRec-score α=0.3", method_sasrec_score, {"alpha": 0.3}),
        ("SASRec-score α=0.5", method_sasrec_score, {"alpha": 0.5}),
        ("SASRec-score α=0.7", method_sasrec_score, {"alpha": 0.7}),
        # RRF
        ("RRF k=10", method_rrf, {"k": 10}),
        ("RRF k=30", method_rrf, {"k": 30}),
        ("RRF k=60", method_rrf, {"k": 60}),
        # Boost-only
        ("Boost-only p70 b=0.05", method_boost_only, {"threshold_pct": 70, "boost": 0.05}),
        ("Boost-only p70 b=0.1", method_boost_only, {"threshold_pct": 70, "boost": 0.1}),
        ("Boost-only p50 b=0.1", method_boost_only, {"threshold_pct": 50, "boost": 0.1}),
        # Conditional
        ("Conditional gap<0.03 α=0.3", method_conditional, {"uncertainty_threshold": 0.03, "alpha": 0.3}),
        ("Conditional gap<0.05 α=0.3", method_conditional, {"uncertainty_threshold": 0.05, "alpha": 0.3}),
        ("Conditional gap<0.10 α=0.3", method_conditional, {"uncertainty_threshold": 0.10, "alpha": 0.3}),
        # Softmax 融合
        ("Softmax-fusion α=0.2 t=1", method_softmax_fusion, {"alpha": 0.2, "gen_temp": 1.0, "sas_temp": 1.0}),
        ("Softmax-fusion α=0.3 t=1", method_softmax_fusion, {"alpha": 0.3, "gen_temp": 1.0, "sas_temp": 1.0}),
        ("Softmax-fusion α=0.5 t=1", method_softmax_fusion, {"alpha": 0.5, "gen_temp": 1.0, "sas_temp": 1.0}),
        # 纯 SASRec（不看生成分数）
        ("Pure SASRec-score", method_sasrec_score, {"alpha": 1.0}),
    ]
    
    print(f"\n{'='*90}", flush=True)
    print(f"重排结果对比", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'方法':<35s}  {'H@1':>7s}  {'H@5':>7s}  {'H@10':>7s}  {'N@5':>7s}  {'N@10':>7s}  {'vs BL':>6s}", flush=True)
    print(f"{'-'*90}", flush=True)
    
    baseline_n10 = None
    best_n10 = 0
    best_method = ""
    
    for method_name, method_fn, params in methods:
        results = []
        
        for idx in range(num_users):
            user_id = t_users[idx]
            bd = beam_data['user_data'][user_id]
            target_item_id = bd['target']
            
            # 构建候选的生成分数
            candidates = bd['candidates']  # list of (item_id, gen_score)
            if len(candidates) == 0:
                results.append([0])
                continue
            
            item_ids = [c[0] for c in candidates]
            gen_scores = np.array([c[1] for c in candidates])
            
            # 获取 SASRec 模型分数 (RecBole item_id = our_id + 1)
            sasrec_scores = sasrec_all_scores[idx][[iid + 1 for iid in item_ids]]
            
            # 重排
            final_scores = method_fn(gen_scores, sasrec_scores=sasrec_scores, **params)
            
            # 排序
            sorted_indices = np.argsort(-final_scores)
            one_results = []
            for si in sorted_indices:
                if item_ids[si] == target_item_id:
                    one_results.append(1)
                else:
                    one_results.append(0)
            results.append(one_results)
        
        m = get_metrics_results(results, metrics)
        total = len(results)
        for k in m:
            m[k] /= total
        
        n10 = m['ndcg@10']
        if baseline_n10 is None:
            baseline_n10 = n10
        
        diff = (n10 - baseline_n10) / baseline_n10 * 100
        marker = " ★" if n10 > best_n10 else ""
        if n10 > best_n10:
            best_n10 = n10
            best_method = method_name
        
        print(f"{method_name:<35s}  {m['hit@1']:7.4f}  {m['hit@5']:7.4f}  {m['hit@10']:7.4f}  {m['ndcg@5']:7.4f}  {m['ndcg@10']:7.4f}  {diff:+5.1f}%{marker}", flush=True)
    
    print(f"\n最佳方法: {best_method} → NDCG@10={best_n10:.4f} ({(best_n10-baseline_n10)/baseline_n10*100:+.1f}%)", flush=True)


def analyze_signal_quality(beam_data, inters, sasrec_all_scores, collab_emb_normed):
    """分析 SASRec 模型分数 vs cosine sim 的区分能力"""
    t_users = beam_data['t_users']
    
    target_sas_scores = []
    nontarget_sas_scores = []
    target_cos_scores = []
    nontarget_cos_scores = []
    target_gen_scores = []
    nontarget_gen_scores = []
    
    for idx in range(len(t_users)):
        user_id = t_users[idx]
        bd = beam_data['user_data'][user_id]
        target_item_id = bd['target']
        candidates = bd['candidates']
        
        if target_item_id == -1 or len(candidates) == 0:
            continue
        
        # 用户 cosine 表征
        items = inters[str(user_id)]
        history = items[:-1][-20:]
        if len(history) == 0:
            continue
        decay = 0.7
        n = len(history)
        weights_arr = np.array([decay ** (n - 1 - j) for j in range(n)])
        weights_arr = weights_arr / weights_arr.sum()
        user_emb = (collab_emb_normed[history] * weights_arr[:, None]).sum(axis=0)
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm
        
        for item_id, gen_score in candidates:
            sas_score = sasrec_all_scores[idx][item_id + 1]
            cos_score = float(collab_emb_normed[item_id] @ user_emb)
            
            if item_id == target_item_id:
                target_sas_scores.append(sas_score)
                target_cos_scores.append(cos_score)
                target_gen_scores.append(gen_score)
            else:
                nontarget_sas_scores.append(sas_score)
                nontarget_cos_scores.append(cos_score)
                nontarget_gen_scores.append(gen_score)
    
    print(f"  样本数: {len(target_sas_scores)} 目标, {len(nontarget_sas_scores)} 非目标")
    
    for name, t_arr, nt_arr in [
        ("生成分数", target_gen_scores, nontarget_gen_scores),
        ("Cosine sim", target_cos_scores, nontarget_cos_scores),
        ("SASRec 模型分数", target_sas_scores, nontarget_sas_scores),
    ]:
        t = np.array(t_arr)
        nt = np.array(nt_arr)
        diff = t.mean() - nt.mean()
        snr = diff / nt.std() if nt.std() > 0 else float('inf')
        print(f"  {name:<15s}: 目标均值={t.mean():.4f}, 非目标均值={nt.mean():.4f}, "
              f"差={diff:.4f}, SNR={snr:.2f}", flush=True)
    
    # SASRec 模型打分做 beam 内排序的效果
    cos_rank_list = []
    sas_rank_list = []
    gen_rank_list = []
    
    for idx in range(len(t_users)):
        user_id = t_users[idx]
        bd = beam_data['user_data'][user_id]
        target_item_id = bd['target']
        candidates = bd['candidates']
        
        if target_item_id == -1 or len(candidates) == 0:
            continue
        
        target_in = False
        for item_id, _ in candidates:
            if item_id == target_item_id:
                target_in = True
                break
        if not target_in:
            continue
        
        items = inters[str(user_id)]
        history = items[:-1][-20:]
        if len(history) == 0:
            continue
        decay = 0.7
        n = len(history)
        weights_arr = np.array([decay ** (n - 1 - j) for j in range(n)])
        weights_arr = weights_arr / weights_arr.sum()
        user_emb = (collab_emb_normed[history] * weights_arr[:, None]).sum(axis=0)
        norm_val = np.linalg.norm(user_emb)
        if norm_val > 0:
            user_emb = user_emb / norm_val
        
        scored = []
        for item_id, gen_score in candidates:
            sas_score = sasrec_all_scores[idx][item_id + 1]
            cos_score = float(collab_emb_normed[item_id] @ user_emb)
            scored.append((item_id, gen_score, sas_score, cos_score))
        
        for sort_key, rank_list in [(1, gen_rank_list), (2, sas_rank_list), (3, cos_rank_list)]:
            by_key = sorted(scored, key=lambda x: x[sort_key], reverse=True)
            for rank, (iid, _, _, _) in enumerate(by_key, 1):
                if iid == target_item_id:
                    rank_list.append(rank)
                    break
    
    print(f"\n  目标 item 在 beam 内的排名 (样本={len(gen_rank_list)}):")
    for name, ranks in [("生成分数", gen_rank_list), ("SASRec模型分数", sas_rank_list), ("Cosine sim", cos_rank_list)]:
        r = np.array(ranks)
        top1 = (r <= 1).sum() / len(r) * 100
        top5 = (r <= 5).sum() / len(r) * 100
        top10 = (r <= 10).sum() / len(r) * 100
        print(f"    {name:<15s}: top1={top1:.1f}%, top5={top5:.1f}%, top10={top10:.1f}%, avg_rank={r.mean():.2f}", flush=True)
    
    # 关键对比: gen vs sasrec 的排名交叉
    gen_r = np.array(gen_rank_list)
    sas_r = np.array(sas_rank_list)
    print(f"\n  SASRec模型 vs 生成分数排名交叉:")
    print(f"    gen<=10 但 sas>10 (损害): {((gen_r <= 10) & (sas_r > 10)).sum()}")
    print(f"    gen>10  但 sas<=10 (收益): {((gen_r > 10) & (sas_r <= 10)).sum()}")


def preload_ensemble_beam(text_save_file, image_save_file, text_c2i, image_c2i, num_beams):
    """预加载 2-path ensemble 候选"""
    td = json.load(open(text_save_file))
    id_ = json.load(open(image_save_file))
    
    t_users = td['all_users']
    num_users = len(t_users)
    
    user_data = {}
    for idx in range(num_users):
        uid = t_users[idx]
        target_code = td['all_targets'][idx].strip().replace(" ", "")
        target_item = text_c2i.get(target_code, -1)
        
        # 合并 text + image beam 为 2-path ensemble 候选
        item_scores = {}
        
        for j in range(num_beams):
            code = td['all_outputs'][idx * num_beams + j].strip().replace(" ", "")
            score = td['all_scores'][idx * num_beams + j]
            item_id = text_c2i.get(code, -1)
            if item_id == -1:
                continue
            if item_id in item_scores:
                item_scores[item_id] = (score + item_scores[item_id]) / 2 + 1
            else:
                item_scores[item_id] = score
        
        i_idx = id_['all_users'].index(uid) if uid in id_['all_users'] else -1
        if i_idx >= 0:
            for j in range(num_beams):
                code = id_['all_outputs'][i_idx * num_beams + j].strip().replace(" ", "")
                score = id_['all_scores'][i_idx * num_beams + j]
                item_id = image_c2i.get(code, -1)
                if item_id == -1:
                    continue
                if item_id in item_scores:
                    item_scores[item_id] = (score + item_scores[item_id]) / 2 + 1
                else:
                    item_scores[item_id] = score
        
        # 转为 list of (item_id, score)
        candidates = [(iid, s) for iid, s in item_scores.items()]
        user_data[uid] = {
            'target': target_item,
            'candidates': candidates,
        }
    
    return {'t_users': t_users, 'user_data': user_data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Instruments")
    parser.add_argument("--output_dir", default="log/Instruments-R1_E1_st-concat-caq")
    parser.add_argument("--index_file", default=".index_lemb_R1_E1_st-concat-caq.json")
    parser.add_argument("--image_index_file", default=".index_vitemb_R1_E1_st-concat-caq.json")
    parser.add_argument("--sasrec_ckpt", default="data/Instruments/saved_sasrec/SASRec-Mar-15-2026_15-23-50.pth")
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    metrics = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
    
    print("加载数据...", flush=True)
    with open(f"data/{args.dataset}/{args.dataset}.inter.json") as f:
        inters = json.load(f)
    
    text_idx = json.load(open(f"data/{args.dataset}/{args.dataset}{args.index_file}"))
    image_idx = json.load(open(f"data/{args.dataset}/{args.dataset}{args.image_index_file}"))
    text_c2i = {(''.join(v)): int(k) for k, v in text_idx.items()}
    image_c2i = {(''.join(v)): int(k) for k, v in image_idx.items()}
    
    text_save = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    image_save = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    
    print("预加载 beam 数据...", flush=True)
    beam_data = preload_ensemble_beam(text_save, image_save, text_c2i, image_c2i, args.num_beams)
    print(f"  用户数: {len(beam_data['t_users'])}", flush=True)
    
    print("加载 SASRec 模型...", flush=True)
    sasrec_scorer = SASRecScorer(args.sasrec_ckpt, device=args.device)
    
    run_rerank_experiment(beam_data, inters, sasrec_scorer, metrics, device=args.device)


if __name__ == '__main__':
    main()
