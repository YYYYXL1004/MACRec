"""
UNGER (TOIS 2025) Stage I — Item Unicodes Generation, faithful (text+collab) re-impl.

Pipeline (paper Sec 3.3):
  E_C : SASRec collaborative embedding   (frozen, 256-d)   <- emb-collab-256.npy
  E_S : Llama text embedding             (frozen, 4096-d)  <- emb-llama-td.npy
  E_T = AdaLN(W E_S + b)                 modality adaption layer (Eq.2), maps E_S -> collab space
  L_align : Info-NCE cross-modality knowledge alignment between E_C and E_T (Eq.3)
  -> integrated embedding = concat(norm(E_C), norm(E_T))   (modality-adaptive fusion)
  -> hierarchical residual k-means -> unified item code  (Unicodes, single branch)

Pragmatic-baseline notes (documented in reproduce_unger_mscgrec.md):
  * DIN -> reuse the SASRec collab embedding (same unified config as CAGRec).
  * Next-item task L_seq is omitted; the adaption layer is trained with L_align only
    (SASRec already provides the collaborative structure).
  * IKD distillation (Stage II) omitted.

Output:
  data/{DS}/{DS}.emb-integrated-unger.npy   integrated embedding [N, 512]
  data/{DS}/{DS}.index_lemb_unger.json      unified single-branch code (prefixes a/b/c/d)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from residual_kmeans import quantize_to_index


class AdaLNAdapter(nn.Module):
    """Modality adaption layer with adaptive LayerNorm (UNGER Eq.2)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        # affine params conditioned on the input itself (AdaLN)
        self.cond = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, e_s):
        h = self.proj(e_s)
        scale, shift = self.cond(e_s).chunk(2, dim=-1)
        return (1 + scale) * self.norm(h) + shift


def info_nce(e_c, e_t, temperature=0.07):
    """In-batch Info-NCE aligning collaborative E_C with adapted semantic E_T."""
    a = F.normalize(e_c, dim=-1)
    b = F.normalize(e_t, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_stage1(e_c, e_s, epochs=200, batch_size=1024, lr=1e-3, temperature=0.07,
                 alpha=1.0, seed=42, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = e_c.shape[0]
    adapter = AdaLNAdapter(e_s.shape[1], e_c.shape[1]).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    e_c_t = torch.tensor(e_c, dtype=torch.float32, device=device)
    e_s_t = torch.tensor(e_s, dtype=torch.float32, device=device)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.numel() < 2:
                continue
            e_t = adapter(e_s_t[idx])
            loss = alpha * info_nce(e_c_t[idx], e_t, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  L_align={total / max(nb, 1):.4f}")
    adapter.eval()
    with torch.no_grad():
        e_t_all = adapter(e_s_t).cpu().numpy()
    return e_t_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--code_num", type=int, default=256)
    p.add_argument("--save_name", type=str, default="unger")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ds = args.dataset
    ddir = os.path.join(args.data_root, ds)
    e_c = np.load(os.path.join(ddir, f"{ds}.emb-collab-256.npy"))
    e_s = np.load(os.path.join(ddir, f"{ds}.emb-llama-td.npy"))
    assert e_c.shape[0] == e_s.shape[0]
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[UNGER Stage I] {ds}: E_C{e_c.shape} E_S{e_s.shape} on {device}")

    e_t = train_stage1(e_c, e_s, epochs=args.epochs, lr=args.lr,
                       temperature=args.temperature, seed=args.seed, device=device)

    # modality-adaptive fusion -> integrated embedding (Unicodes input)
    def l2(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    integrated = np.concatenate([l2(e_c), l2(e_t)], axis=1).astype(np.float32)
    int_path = os.path.join(ddir, f"{ds}.emb-integrated-unger.npy")
    np.save(int_path, integrated)
    print(f"[UNGER Stage I] integrated emb {integrated.shape} -> {int_path}")

    out_path = os.path.join(ddir, f"{ds}.index_lemb_{args.save_name}.json")
    print(f"[UNGER Stage I] quantizing integrated embedding -> unified code")
    quantize_to_index(integrated, out_path, prefixes="abcd",
                      n_levels=3, code_num=args.code_num, seed=args.seed, l2norm=False)


if __name__ == "__main__":
    main()
