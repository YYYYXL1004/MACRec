"""
Motivation 实证验证: Content Similarity ≠ Collaborative Similarity

四项分析:
(a) 相似度散点图 + Spearman 相关系数
(b) 邻居重叠度 (Jaccard Index)
(c) Code Space 协同盲区分析
(d) t-SNE 双空间可视化

用法:
    cd MACRec
    python analysis/motivation_analysis.py --dataset Instruments
"""

import argparse
import json
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

# 论文级别绘图参数
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--text_suffix', type=str, default='.emb-llama-td.npy')
    parser.add_argument('--image_suffix', type=str, default='.emb-ViT-L-14.npy')
    parser.add_argument('--collab_suffix', type=str, default='.emb-collab-256.npy')
    parser.add_argument('--index_file', type=str, default='.index_lemb_Inst3.15.json')
    parser.add_argument('--n_pairs', type=int, default=50000,
                        help='散点图采样的 item pair 数量')
    parser.add_argument('--topk', type=int, default=10,
                        help='邻居重叠分析的 K')
    parser.add_argument('--output_dir', type=str, default='./analysis/figures')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_embeddings(args):
    """加载三路向量并 L2 归一化"""
    ds = args.dataset
    base = os.path.join(args.data_dir, ds)

    text = np.load(os.path.join(base, ds + args.text_suffix)).astype(np.float32)
    image = np.load(os.path.join(base, ds + args.image_suffix)).astype(np.float32)
    collab = np.load(os.path.join(base, ds + args.collab_suffix)).astype(np.float32)

    text_norm = normalize(text, norm='l2', axis=1)
    image_norm = normalize(image, norm='l2', axis=1)
    collab_norm = normalize(collab, norm='l2', axis=1)

    print(f"加载完成: text={text.shape}, image={image.shape}, collab={collab.shape}")
    return text_norm, image_norm, collab_norm


# ============================================================
# (a) 相似度散点图 + Spearman 相关系数
# ============================================================
def analysis_scatter(text_norm, image_norm, collab_norm, args):
    """采样 item pairs，计算 content sim vs collab sim"""
    np.random.seed(args.seed)
    n_items = text_norm.shape[0]
    n_pairs = args.n_pairs

    # 随机采样不重复的 pair
    idx_i = np.random.randint(0, n_items, size=n_pairs)
    idx_j = np.random.randint(0, n_items, size=n_pairs)
    # 去掉 i==j 的 pair
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    # 向量化计算 cosine similarity (已归一化，点积即可)
    text_sim = np.sum(text_norm[idx_i] * text_norm[idx_j], axis=1)
    image_sim = np.sum(image_norm[idx_i] * image_norm[idx_j], axis=1)
    collab_sim = np.sum(collab_norm[idx_i] * collab_norm[idx_j], axis=1)

    # Spearman 相关系数
    rho_text, p_text = spearmanr(text_sim, collab_sim)
    rho_image, p_image = spearmanr(image_sim, collab_sim)

    print("\n=== (a) Content vs Collaborative Similarity ===")
    print(f"采样 {len(idx_i)} 对 item pairs")
    print(f"Text  vs Collab: Spearman ρ = {rho_text:.4f} (p = {p_text:.2e})")
    print(f"Image vs Collab: Spearman ρ = {rho_image:.4f} (p = {p_image:.2e})")

    # 绘图: 两个子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # 采样绘图点 (太密就只画一部分)
    plot_n = min(5000, len(idx_i))
    plot_idx = np.random.choice(len(idx_i), plot_n, replace=False)

    for ax, c_sim, label, rho in [
        (axes[0], text_sim, 'Text', rho_text),
        (axes[1], image_sim, 'Image', rho_image),
    ]:
        ax.scatter(c_sim[plot_idx], collab_sim[plot_idx],
                   s=3, alpha=0.15, c='#4C72B0', edgecolors='none')
        ax.set_xlabel(f'{label} Cosine Similarity')
        ax.set_ylabel('Collaborative Cosine Similarity')
        ax.set_title(f'{label} vs Collaborative (ρ = {rho:.3f})')
        ax.set_xlim(-0.3, 1.05)
        ax.set_ylim(-0.3, 1.05)
        # 添加对角线参考
        ax.plot([-0.3, 1.05], [-0.3, 1.05], 'r--', alpha=0.4, linewidth=1)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'motivation_scatter.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存: {out_path}")

    return rho_text, rho_image


# ============================================================
# (b) 邻居重叠度 (Jaccard Index)
# ============================================================
def analysis_neighbor_overlap(text_norm, image_norm, collab_norm, args):
    """计算 content top-K 和 collab top-K 邻居的 Jaccard 重叠度"""
    K = args.topk
    n_items = text_norm.shape[0]

    # 批量计算相似度矩阵 (如果 item 太多就分块)
    print(f"\n=== (b) Neighbor Overlap (K={K}) ===")
    print(f"计算 {n_items} x {n_items} 相似度矩阵...")

    text_sim_mat = text_norm @ text_norm.T
    image_sim_mat = image_norm @ image_norm.T
    collab_sim_mat = collab_norm @ collab_norm.T

    # 取 top-K 邻居 (排除自己)
    np.fill_diagonal(text_sim_mat, -1)
    np.fill_diagonal(image_sim_mat, -1)
    np.fill_diagonal(collab_sim_mat, -1)

    text_topk = np.argpartition(-text_sim_mat, K, axis=1)[:, :K]
    image_topk = np.argpartition(-image_sim_mat, K, axis=1)[:, :K]
    collab_topk = np.argpartition(-collab_sim_mat, K, axis=1)[:, :K]

    # Jaccard index
    jaccard_text = []
    jaccard_image = []
    for i in range(n_items):
        t_set = set(text_topk[i])
        img_set = set(image_topk[i])
        c_set = set(collab_topk[i])

        jt = len(t_set & c_set) / len(t_set | c_set) if len(t_set | c_set) > 0 else 0
        ji = len(img_set & c_set) / len(img_set | c_set) if len(img_set | c_set) > 0 else 0
        jaccard_text.append(jt)
        jaccard_image.append(ji)

    jaccard_text = np.array(jaccard_text)
    jaccard_image = np.array(jaccard_image)

    print(f"Text  vs Collab: Jaccard 均值={jaccard_text.mean():.4f}, 中位数={np.median(jaccard_text):.4f}")
    print(f"Image vs Collab: Jaccard 均值={jaccard_image.mean():.4f}, 中位数={np.median(jaccard_image):.4f}")
    print(f"Jaccard=0 的比例: text={np.mean(jaccard_text == 0):.2%}, image={np.mean(jaccard_image == 0):.2%}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, jac, label in [
        (axes[0], jaccard_text, 'Text'),
        (axes[1], jaccard_image, 'Image'),
    ]:
        ax.hist(jac, bins=30, color='#4C72B0', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axvline(jac.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean = {jac.mean():.3f}')
        ax.set_xlabel('Jaccard Index')
        ax.set_ylabel('Number of Items')
        ax.set_title(f'{label} vs Collaborative Neighbors (K={K})')
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'motivation_overlap.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存: {out_path}")

    return jaccard_text.mean(), jaccard_image.mean()


# ============================================================
# (c) Code Space 协同盲区分析
# ============================================================
def analysis_codespace(collab_norm, args):
    """分析 baseline RQVAE codes 是否反映协同结构"""
    ds = args.dataset
    index_path = os.path.join(args.data_dir, ds, ds + args.index_file)

    if not os.path.exists(index_path):
        print(f"\n=== (c) Code Space 分析: 跳过 (文件不存在: {index_path}) ===")
        return None, None

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    # 按 first-level code 分组
    code_groups = defaultdict(list)
    for item_id_str, codes in index_data.items():
        first_code = codes[0]
        code_groups[first_code].append(int(item_id_str))

    # 计算每个 code group 内的平均 collab similarity
    n_items = collab_norm.shape[0]
    group_sims = []
    for code, items in code_groups.items():
        # 过滤有效 item id
        valid_items = [i for i in items if i < n_items]
        if len(valid_items) < 2:
            continue
        # 簇内所有 pair 的 collab sim
        embs = collab_norm[valid_items]
        sim_mat = embs @ embs.T
        np.fill_diagonal(sim_mat, 0)
        n = len(valid_items)
        avg_sim = sim_mat.sum() / (n * (n - 1)) if n > 1 else 0
        group_sims.append(avg_sim)

    # 随机 pair 的 collab sim
    np.random.seed(args.seed)
    rand_i = np.random.randint(0, n_items, size=10000)
    rand_j = np.random.randint(0, n_items, size=10000)
    mask = rand_i != rand_j
    rand_sim = np.sum(collab_norm[rand_i[mask]] * collab_norm[rand_j[mask]], axis=1)

    group_sims = np.array(group_sims)

    print(f"\n=== (c) Code Space 协同盲区分析 ===")
    print(f"Code groups 数量: {len(group_sims)}")
    print(f"Same-code 簇内 collab sim 均值: {group_sims.mean():.4f} ± {group_sims.std():.4f}")
    print(f"Random pair collab sim 均值:     {rand_sim.mean():.4f} ± {rand_sim.std():.4f}")
    print(f"差异: {group_sims.mean() - rand_sim.mean():.4f}")

    # 绘图: box plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bp = ax.boxplot([group_sims, rand_sim],
                    tick_labels=['Same First-Level\nCode Items', 'Random\nItem Pairs'],
                    patch_artist=True,
                    widths=0.5,
                    showfliers=False)
    bp['boxes'][0].set_facecolor('#4C72B0')
    bp['boxes'][1].set_facecolor('#DD8452')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)

    ax.set_ylabel('Collaborative Cosine Similarity')
    ax.set_title('Code Space vs Collaborative Structure')

    # 添加均值标注
    ax.text(1, group_sims.mean() + 0.02, f'μ={group_sims.mean():.3f}',
            ha='center', fontsize=10, color='#4C72B0', fontweight='bold')
    ax.text(2, rand_sim.mean() + 0.02, f'μ={rand_sim.mean():.3f}',
            ha='center', fontsize=10, color='#DD8452', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'motivation_codespace.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存: {out_path}")

    return group_sims.mean(), rand_sim.mean()


# ============================================================
# (d) t-SNE 双空间可视化
# ============================================================
def analysis_tsne(text_norm, collab_norm, args):
    """t-SNE 可视化: 用 collab clusters 着色，对比 text space vs collab space"""
    n_items = text_norm.shape[0]
    n_clusters = 8

    print(f"\n=== (d) t-SNE 双空间可视化 ===")
    print(f"KMeans(K={n_clusters}) on collab embeddings...")
    km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    collab_labels = km.fit_predict(collab_norm)

    # 如果 item 太多，采样
    max_points = 3000
    if n_items > max_points:
        np.random.seed(args.seed)
        sample_idx = np.random.choice(n_items, max_points, replace=False)
    else:
        sample_idx = np.arange(n_items)

    text_sample = text_norm[sample_idx]
    collab_sample = collab_norm[sample_idx]
    labels_sample = collab_labels[sample_idx]

    print(f"t-SNE on text space ({len(sample_idx)} points)...")
    tsne_text = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)
    text_2d = tsne_text.fit_transform(text_sample)

    print(f"t-SNE on collab space ({len(sample_idx)} points)...")
    tsne_collab = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)
    collab_2d = tsne_collab.fit_transform(collab_sample)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cmap = plt.cm.get_cmap('tab10', n_clusters)

    for ax, emb_2d, title in [
        (axes[0], text_2d, 'Text Feature Space'),
        (axes[1], collab_2d, 'Collaborative Space'),
    ]:
        for c in range(n_clusters):
            mask = labels_sample == c
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       s=5, alpha=0.5, c=[cmap(c)], label=f'C{c}', edgecolors='none')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # 单独的 legend
    handles, lbs = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc='center right', title='Collab\nCluster',
              bbox_to_anchor=(1.0, 0.5), markerscale=3)
    fig.suptitle('Items colored by Collaborative Clusters', y=1.02, fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'motivation_tsne.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存: {out_path}")


# ============================================================
# (e) Figure 1: Content Space vs Collaborative Space
# ============================================================
def _fit_cluster_ellipse(points_2d, n_std=1.8):
    """根据 2D 点集拟合椭圆参数 (中心, 宽, 高, 角度)"""
    if len(points_2d) < 3:
        return None
    mean = points_2d.mean(axis=0)
    cov = np.cov(points_2d, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 从大到小排列
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    return mean, width, height, angle


def analysis_fig1_content_vs_collab(text_norm, image_norm, collab_norm, args):
    """
    Figure 1 (2×2): 品牌视角 + 类别视角, content vs collaborative
    Row 1: (a) Text by brand  →  (b) Collab by brand
    Row 2: (c) Image by category  →  (d) Collab by category
    """
    print("\n=== (e) Figure 1: 2×2 content vs collaborative ===")

    # --- 1. 加载 item 元数据 ---
    ds = args.dataset
    item_path = os.path.join(args.data_dir, ds, ds + '.item.json')
    with open(item_path, 'r') as f:
        items_meta = json.load(f)

    brands = {}
    categories = {}
    for k, v in items_meta.items():
        idx = int(k)
        if idx >= text_norm.shape[0]:
            continue
        brands[idx] = v.get('brand', '')
        raw_cats = v.get('categories', '').replace('&amp;', '&').split(',')
        categories[idx] = raw_cats[1].strip() if len(raw_cats) >= 2 else raw_cats[0].strip()

    # --- 2. 筛选品牌 item ---
    target_brands = ['Behringer', 'JIM DUNLOP', 'Fender']
    brand_colors = {'Behringer': '#1F77B4', 'JIM DUNLOP': '#FF7F0E', 'Fender': '#2CA02C'}
    brand_alias = {'Behringer': 'Brand A', 'JIM DUNLOP': 'Brand B', 'Fender': 'Brand C'}

    brand_indices = []
    brand_labels = []
    for idx in sorted(brands.keys()):
        if brands[idx] in target_brands:
            brand_indices.append(idx)
            brand_labels.append(brands[idx])
    brand_indices = np.array(brand_indices)
    brand_labels = np.array(brand_labels)
    print(f"品牌 items: {len(brand_indices)} ({', '.join(f'{b}={int((brand_labels==b).sum())}' for b in target_brands)})")

    # --- 3. 筛选类别 item ---
    target_cats = ['Guitars', 'Amplifiers & Effects', 'Microphones & Accessories']
    cat_colors = {'Guitars': '#1F77B4', 'Amplifiers & Effects': '#FF7F0E', 'Microphones & Accessories': '#D62728'}
    cat_alias = {'Guitars': 'Category A', 'Amplifiers & Effects': 'Category B', 'Microphones & Accessories': 'Category C'}
    cat_short = {'Guitars': 'Guitars', 'Amplifiers & Effects': 'Amplifiers', 'Microphones & Accessories': 'Microphones'}

    cat_indices = []
    cat_labels = []
    for idx in sorted(categories.keys()):
        if categories[idx] in target_cats:
            cat_indices.append(idx)
            cat_labels.append(categories[idx])
    cat_indices = np.array(cat_indices)
    cat_labels = np.array(cat_labels)
    print(f"类别 items: {len(cat_indices)} ({', '.join(f'{cat_short[c]}={int((cat_labels==c).sum())}' for c in target_cats)})")

    # --- 4. t-SNE (4组) ---
    tsne_kw = dict(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)

    print("t-SNE: text (brands)...")
    text_brand_2d = TSNE(**tsne_kw).fit_transform(text_norm[brand_indices])
    print("t-SNE: collab (brands)...")
    collab_brand_2d = TSNE(**tsne_kw).fit_transform(collab_norm[brand_indices])
    print("t-SNE: image (categories)...")
    image_cat_2d = TSNE(**tsne_kw).fit_transform(image_norm[cat_indices])
    print("t-SNE: collab (categories)...")
    collab_cat_2d = TSNE(**tsne_kw).fit_transform(collab_norm[cat_indices])

    # --- 5. 绘图 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))

    def _draw_panel(ax, emb_2d, labels, groups, color_map, alias_map, short_map,
                    draw_ellipse=True):
        """在 ax 上画散点 + 可选椭圆标注"""
        for g in groups:
            mask = labels == g
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                s=12, alpha=0.7, c=color_map[g],
                label=short_map[g] if short_map else g,
                edgecolors='none'
            )
        # 椭圆标注 (仅 content 面板)
        if draw_ellipse:
            for g in groups:
                mask = labels == g
                pts = emb_2d[mask]
                result = _fit_cluster_ellipse(pts, n_std=2.0)
                if result is None:
                    continue
                center, w, h, angle = result
                ell = Ellipse(
                    xy=center, width=w, height=h, angle=angle,
                    fill=False, edgecolor='red', linewidth=2.5,
                    linestyle='--', zorder=8
                )
                ax.add_patch(ell)
                ax.text(
                    center[0], center[1] + h * 0.6,
                    alias_map[g], fontsize=13, fontweight='bold',
                    ha='center', va='bottom', color='black', zorder=10
                )
        ax.legend(loc='upper left', fontsize=9, frameon=True,
                  fancybox=True, framealpha=0.9, markerscale=1.5)

    # (a) Text by brand
    _draw_panel(axes[0, 0], text_brand_2d, brand_labels, target_brands,
                brand_colors, brand_alias, {b: b for b in target_brands},
                draw_ellipse=True)
    axes[0, 0].set_xlabel('Text Embeddings', fontsize=12, fontweight='bold')

    # (b) Collab by brand — 品牌在协同空间中混杂
    _draw_panel(axes[0, 1], collab_brand_2d, brand_labels, target_brands,
                brand_colors, brand_alias, {b: b for b in target_brands},
                draw_ellipse=False)
    axes[0, 1].set_xlabel('Collaborative Embeddings', fontsize=12, fontweight='bold')

    # (c) Image by category
    _draw_panel(axes[1, 0], image_cat_2d, cat_labels, target_cats,
                cat_colors, cat_alias, cat_short,
                draw_ellipse=True)
    axes[1, 0].set_xlabel('Image Embeddings', fontsize=12, fontweight='bold')

    # (d) Collab by category — 类别在协同空间中混杂
    _draw_panel(axes[1, 1], collab_cat_2d, cat_labels, target_cats,
                cat_colors, cat_alias, cat_short,
                draw_ellipse=False)
    axes[1, 1].set_xlabel('Collaborative Embeddings', fontsize=12, fontweight='bold')

    # 统一样式
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    # --- 保存 ---
    paper_dir = os.path.join(os.path.dirname(args.data_dir), '..', 'paper')
    paper_dir = os.path.normpath(paper_dir)
    os.makedirs(paper_dir, exist_ok=True)

    for ext in ['pdf', 'png']:
        out_path = os.path.join(paper_dir, f'fig1_content_vs_collab.{ext}')
        fig.savefig(out_path)
        print(f"已保存: {out_path}")

    out_path_analysis = os.path.join(args.output_dir, 'fig1_content_vs_collab.pdf')
    fig.savefig(out_path_analysis)
    print(f"已保存: {out_path_analysis}")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    text_norm, image_norm, collab_norm = load_embeddings(args)

    # (a) 散点图
    rho_t, rho_i = analysis_scatter(text_norm, image_norm, collab_norm, args)

    # (b) 邻居重叠
    jac_t, jac_i = analysis_neighbor_overlap(text_norm, image_norm, collab_norm, args)

    # (c) Code space
    cs_same, cs_rand = analysis_codespace(collab_norm, args)

    # (d) t-SNE
    analysis_tsne(text_norm, collab_norm, args)

    # (e) Figure 1: Content vs Collaborative
    analysis_fig1_content_vs_collab(text_norm, image_norm, collab_norm, args)

    # 汇总
    print("\n" + "=" * 60)
    print("Motivation 验证汇总")
    print("=" * 60)
    print(f"(a) Spearman ρ:  text-collab={rho_t:.4f}, image-collab={rho_i:.4f}")
    print(f"(b) Jaccard 均值: text={jac_t:.4f}, image={jac_i:.4f}")
    if cs_same is not None:
        print(f"(c) Code space:  same-code={cs_same:.4f}, random={cs_rand:.4f}, gap={cs_same - cs_rand:.4f}")
    print(f"(d) t-SNE 图已生成")
    print(f"\n所有图表保存在: {args.output_dir}/")


if __name__ == '__main__':
    main()
