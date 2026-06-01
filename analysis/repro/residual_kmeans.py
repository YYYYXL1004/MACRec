"""
Hierarchical residual K-means quantization (UNGER Algorithm 1 style).

Quantizes an item embedding matrix [N, D] into 3 residual k-means levels
(code_num clusters each) plus a 4th uniquifying level (collision counter),
producing the same 4-token index format used by CAGRec's RQ-VAE codes:
    {item_id: ["<a_5>", "<b_12>", "<c_3>", "<d_8>"], ...}

Shared by both baselines:
  * UNGER   -> quantize the fused text+collab integrated embedding (prefixes a/b/c/d)
  * MSCGRec -> quantize the SASRec collaborative embedding as a separate modality
              (prefixes P/Q/R/S)

Usage:
  python residual_kmeans.py --emb_path data/Instruments/Instruments.emb-collab-256.npy \
      --out_path data/Instruments/Instruments.index_collab_mscgrec.json \
      --prefixes PQRS --n_levels 3 --code_num 256 --seed 42
"""
import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def residual_kmeans(emb, n_levels=3, code_num=256, seed=42):
    """Return an [N, n_levels] int array of residual k-means cluster ids."""
    residual = emb.astype(np.float64).copy()
    codes = np.zeros((emb.shape[0], n_levels), dtype=np.int64)
    for level in range(n_levels):
        k = min(code_num, residual.shape[0])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(residual)
        codes[:, level] = labels
        # subtract assigned centroid to form the next-level residual (Alg.1 Eq.9)
        residual = residual - km.cluster_centers_[labels]
        print(f"  level {level}: k={k}, residual norm mean={np.linalg.norm(residual, axis=1).mean():.4f}")
    return codes


def add_unique_level(codes, code_num=256):
    """Append a 4th level that disambiguates items sharing the same 3-level prefix.

    Guarantees item uniqueness (required for retrieval). Warns if any collision
    bucket exceeds code_num (would overflow a single token); in practice buckets
    are small.
    """
    buckets = {}
    last = np.zeros((codes.shape[0],), dtype=np.int64)
    max_bucket = 0
    for i in range(codes.shape[0]):
        key = tuple(codes[i].tolist())
        c = buckets.get(key, 0)
        last[i] = c
        buckets[key] = c + 1
        max_bucket = max(max_bucket, c + 1)
    if max_bucket > code_num:
        print(f"  [WARN] max collision bucket {max_bucket} > code_num {code_num}; "
              f"increase n_levels or code_num.")
    print(f"  uniquify: {len(buckets)} distinct 3-level prefixes, "
          f"max bucket size {max_bucket}")
    return np.concatenate([codes, last[:, None]], axis=1)


def write_index(codes, prefixes, out_path):
    assert codes.shape[1] == len(prefixes), "prefix count must match code levels"
    index = {}
    for i in range(codes.shape[0]):
        index[str(i)] = [f"<{prefixes[l]}_{int(codes[i, l])}>" for l in range(codes.shape[1])]
    n_unique = len({"".join(v) for v in index.values()})
    assert n_unique == codes.shape[0], f"non-unique codes: {n_unique}/{codes.shape[0]}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"  saved {codes.shape[0]} unique codes -> {out_path}")


def quantize_to_index(emb, out_path, prefixes="abcd", n_levels=3, code_num=256, seed=42, l2norm=True):
    """High-level helper used both from CLI and from other repro scripts."""
    if l2norm:
        emb = normalize(emb, norm="l2", axis=1)
    codes = residual_kmeans(emb, n_levels=n_levels, code_num=code_num, seed=seed)
    codes = add_unique_level(codes, code_num=code_num)
    write_index(codes, prefixes, out_path)
    return codes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--prefixes", type=str, default="abcd",
                   help="4 chars, one per level incl. uniquify level (e.g. abcd / ABCD / PQRS)")
    p.add_argument("--n_levels", type=int, default=3, help="residual k-means levels before uniquify")
    p.add_argument("--code_num", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_l2norm", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert len(args.prefixes) == args.n_levels + 1, \
        f"--prefixes must have n_levels+1={args.n_levels + 1} chars"
    emb = np.load(args.emb_path)
    print(f"loaded {args.emb_path} shape={emb.shape}")
    quantize_to_index(emb, args.out_path, prefixes=args.prefixes,
                      n_levels=args.n_levels, code_num=args.code_num,
                      seed=args.seed, l2norm=not args.no_l2norm)
