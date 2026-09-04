import torch
import torch.nn.functional as F

@torch.no_grad()
def check_collapse(emb_path: str, sample_pairs: int = 200_000, device: str = "cpu"):
    payload = torch.load(emb_path, map_location="cpu")
    emb = payload["emb"].to(device)              # [N, d]
    N, d = emb.shape
    print(f"[load] N={N:,}, d={d}, normalize_flag={payload.get('normalize', None)}")

    # ---- A) norm stats (raw) ----
    norms = emb.norm(dim=-1)                     # [N]
    norm_mean = norms.mean().item()
    norm_std  = norms.std(unbiased=False).item()
    print("\n[A] Norm stats (raw emb)")
    print(f"  mean={norm_mean:.6f} std={norm_std:.6f} std/mean={(norm_std/(norm_mean+1e-12)):.6e}")
    print(f"  min={norms.min().item():.6f} max={norms.max().item():.6f}")

    # ---- B) spread: per-dim std ----
    dim_std = emb.std(dim=0, unbiased=False)     # [d]
    print("\n[B] Dim-wise std (raw emb)")
    print(f"  mean(dim_std)={dim_std.mean().item():.6f}  "
          f"min(dim_std)={dim_std.min().item():.6f}  max(dim_std)={dim_std.max().item():.6f}")

    # normalize for cosine analysis
    embn = F.normalize(emb, p=2, dim=-1)

    # ---- C) random pair cosine distribution ----
    # sample random pairs (i,j) and compute cosine = dot
    m = min(sample_pairs, N * (N - 1) // 2)  # safe cap
    i = torch.randint(0, N, (m,), device=device)
    j = torch.randint(0, N, (m,), device=device)
    mask = (i != j)
    i, j = i[mask], j[mask]
    cos = (embn[i] * embn[j]).sum(dim=-1)

    print("\n[C] Cosine similarity on random pairs (normalized emb)")
    print(f"  mean={cos.mean().item():.6f} std={cos.std(unbiased=False).item():.6f}")
    q = torch.quantile(cos, torch.tensor([0.01, 0.5, 0.99], device=device)).tolist()
    print(f"  p01={q[0]:.6f} median={q[1]:.6f} p99={q[2]:.6f}")
    print(f"  min={cos.min().item():.6f} max={cos.max().item():.6f}")

    # ---- D) Approx top-1 cosine to random candidate pool (exclude self) ----
    Q = min(200, N)
    q_idx = torch.randperm(N, device=device)[:Q]

    # cand = all nodes (or a random subset), but we'll mask out self
    Cemb = embn  # [N, d]
    Qemb = embn[q_idx]  # [Q, d]
    scores = Qemb @ Cemb.T  # [Q, N]

    # exclude self by setting its score to -inf
    scores[torch.arange(Q, device=device), q_idx] = -1e9
    top1 = scores.max(dim=1).values
    print(f"top1 mean={top1.mean().item():.6f} std={top1.std(unbiased=False).item():.6f} "
          f"min={top1.min().item():.6f} max={top1.max().item():.6f}")

    # ---- quick heuristics ----
    print("\n[Heuristic flags]")
    if norm_mean < 1e-3:
        print("  ⚠️ norms very small -> possible collapse to near-zero.")
    if (norm_std / (norm_mean + 1e-12)) < 1e-3:
        print("  ⚠️ norms have extremely low variance -> possible constant-vector collapse.")
    if cos.mean().item() > 0.8 and cos.std(unbiased=False).item() < 0.05:
        print("  ⚠️ random-pair cosines are very high and tight -> possible directional collapse / anisotropy.")
    else:
        print("  ✅ no obvious collapse signal from simple statistics (still check downstream behavior).")

if __name__ == "__main__":
    check_collapse("../ckpts/zone_embeddings.pt", device="cpu")
