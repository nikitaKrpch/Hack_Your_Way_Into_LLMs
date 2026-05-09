"""Probe training: linear / MLP / single- and 4-head attention probes.
5 seeds per arch, two regimes (batch / incremental), saves probe weights.

Multi-layer features: probes can use multiple residual layers via concatenation,
mean/max pooling, or learned layer-wise attention.

Usage:
    python train_probe.py \
        --extracts_dir ./extracts/gemma4_31b \
        --manifest    ./extracts/gemma4_31b/extraction_metadata.json \
        --out_dir     ./probes \
        --task        refusal_gemma4_31b   # or cyber_*

The extracts directory should contain per-sample .pt files produced by
extract_residuals.py. Manifest is the JSON written alongside extracts.
"""
import os, sys, json, time, math, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEEDS = [0, 1, 2, 3, 4]
ARCHS = ["linear", "mlp", "attention", "attention_4h",
        "linear_ml", "mlp_ml", "layer_attn"]  # multi-layer variants
REGIMES = ["batch", "incremental"]


# ---------------- probe modules ----------------
class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x_final, x_full=None, mask=None):
        return self.fc(x_final).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
    def forward(self, x_final, x_full=None, mask=None):
        return self.net(x_final).squeeze(-1)


class AttentionProbe(nn.Module):
    """Single learned-query attention over tokens."""
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
    def forward(self, x_final, x_full, mask):
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)                    # (B, N)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)   # (B, d)
        return self.head(pooled).squeeze(-1), alpha


class MultiHeadAttentionProbe(nn.Module):
    """K learned queries; concat per-head pooled vectors."""
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_heads, d) / math.sqrt(d))
        self.head = nn.Linear(d * n_heads, 1)
        self.n_heads = n_heads
    def forward(self, x_final, x_full, mask):
        B, N, d = x_full.shape
        logits = torch.einsum("bnd,kd->bnk", x_full, self.q) / math.sqrt(d)  # (B,N,K)
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(logits, dim=1)                       # (B, N, K)
        pooled = torch.einsum("bnk,bnd->bkd", alpha, x_full)   # (B, K, d)
        pooled = pooled.reshape(B, -1)                          # (B, K*d)
        return self.head(pooled).squeeze(-1), alpha


class LinearMultiLayerProbe(nn.Module):
    """Linear probe over multiple layers: concatenate layer outputs then project."""
    def __init__(self, d, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.fc = nn.Linear(d * n_layers, 1)
    def forward(self, x_final, x_full, mask):
        # x_full expected to be (B, n_layers, N, d)
        # Use only last token per layer to avoid padding issues
        B, nL, N, d = x_full.shape
        last = x_full[:, :, -1, :]  # (B, n_layers, d)
        x = last.reshape(B, nL * d)
        return self.fc(x).squeeze(-1)


class MLPMultiLayerProbe(nn.Module):
    """MLP probe over multiple layers with hidden layer."""
    def __init__(self, d, n_layers, hidden=256, drop=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.net = nn.Sequential(
            nn.Linear(d * n_layers, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))
    def forward(self, x_final, x_full, mask):
        B, nL, N, d = x_full.shape
        last = x_full[:, :, -1, :]  # (B, n_layers, d)
        x = last.reshape(B, nL * d)
        return self.net(x).squeeze(-1)


class LayerAttentionProbe(nn.Module):
    """Attention over layers to produce weighted combination."""
    def __init__(self, d, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
        self.scale = nn.Parameter(torch.ones(n_layers))
        self.shift = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x_final, x_full, mask):
        # x_full: (B, n_layers, N, d) - last token per layer
        B, nL, N, d = x_full.shape
        last_per_layer = x_full[:, :, -1, :]  # (B, n_layers, d)
        last_per_layer = last_per_layer * self.scale.view(1, nL, 1) + self.shift.view(1, nL, 1)
        logits = (last_per_layer @ self.q) / math.sqrt(d)
        alpha = F.softmax(logits, dim=1)
        weighted = torch.einsum("bl,bld->bd", alpha, last_per_layer)
        return self.head(weighted).squeeze(-1), alpha


def make_probe(arch, d, n_layers=1):
    if arch == "linear":      return LinearProbe(d)
    if arch == "mlp":         return MLPProbe(d)
    if arch == "attention":   return AttentionProbe(d)
    if arch == "attention_4h":return MultiHeadAttentionProbe(d, n_heads=4)
    if arch == "linear_ml":   return LinearMultiLayerProbe(d, n_layers)
    if arch == "mlp_ml":      return MLPMultiLayerProbe(d, n_layers)
    if arch == "layer_attn":  return LayerAttentionProbe(d, n_layers)
    raise ValueError(arch)


# ---------------- data loading ----------------
def load_extract(extracts_dir, sample_id):
    return torch.load(str(extracts_dir / f"{sample_id}.pt"), weights_only=False)


def get_full_tokens(ex):
    """Pick the per-token residual tensor regardless of which key was used.

    extract_residuals.py writes 'residuals' (n_layers_selected, N, d). Older
    extracts may have 'middle_layer_all_tokens' (N, d). Both supported.
    """
    if "residuals" in ex:
        r = ex["residuals"]
        # If multi-layer, return all layers for multi-layer probes
        if r.dim() == 3 and r.shape[0] > 1:
            return r      # (n_layers, N, d)
        return r.squeeze(0) if r.dim() == 3 else r
    if "middle_layer_all_tokens" in ex:
        return ex["middle_layer_all_tokens"]
    raise KeyError(f"extract missing residuals key (looked for 'residuals' / 'middle_layer_all_tokens')")


def build_dataset(samples, label_fn, extracts_dir):
    x_final, x_full_list, mask_list, y, ids = [], [], [], [], []
    x_multi_layer_list = []  # Store all layers for multi-layer probes
    skipped = 0
    for s in samples:
        try:
            ex = load_extract(extracts_dir, s["sample_id"])
        except FileNotFoundError:
            skipped += 1; continue
        full = get_full_tokens(ex).to(DTYPE)
        m = ex["attention_mask"]
        if not m.any():
            skipped += 1; continue
        last = int(m.nonzero().max().item())
        if full.dim() == 3 and full.shape[0] > 1:
            x_final.append(full[0, last])  # first layer, last token
        else:
            x_final.append(full[last])
        x_full_list.append(full)
        mask_list.append(m)
        y.append(label_fn(s))
        ids.append(s["sample_id"])
        # Store all layers for multi-layer probes
        if full.dim() == 3 and full.shape[0] > 1:
            x_multi_layer_list.append(full)
        else:
            # Single layer - expand to match expected format
            x_multi_layer_list.append(full.unsqueeze(0))
    if skipped: print(f"    [warn] skipped {skipped}")
    if not x_final: return None
    return {
        "x_final": torch.stack(x_final, dim=0).to(DEVICE),
        "x_full_list": x_full_list,
        "mask_list": mask_list,
        "x_multi_layer_list": x_multi_layer_list,
        "y": torch.tensor(y, dtype=torch.float32, device=DEVICE),
        "ids": ids,
    }


def pad_full(x_full_list, mask_list):
    N = len(x_full_list)
    T_max = max(t.shape[0] for t in x_full_list)
    d = x_full_list[0].shape[1]
    px = torch.zeros(N, T_max, d, dtype=DTYPE, device=DEVICE)
    pm = torch.zeros(N, T_max, dtype=torch.bool, device=DEVICE)
    for i, (t, m) in enumerate(zip(x_full_list, mask_list)):
        T_i = t.shape[0]
        px[i, :T_i] = t.to(DEVICE)
        pm[i, :T_i] = m.to(DEVICE)
    return px, pm


def pad_multi_layer(x_multi_layer_list):
    """Pad multi-layer residuals to same (n_layers, N, d)."""
    N = len(x_multi_layer_list)
    nL = x_multi_layer_list[0].shape[0]
    T_max = max(t.shape[1] for t in x_multi_layer_list)
    d = x_multi_layer_list[0].shape[2]
    px = torch.zeros(N, nL, T_max, d, dtype=DTYPE, device=DEVICE)
    for i, t in enumerate(x_multi_layer_list):
        T_i = t.shape[1]
        px[i, :, :T_i, :] = t.to(DEVICE)
    return px


def pad_multi_layer_last(x_multi_layer_list):
    """For probes that only use last token - output (N, n_layers, 1, d)."""
    N = len(x_multi_layer_list)
    nL = x_multi_layer_list[0].shape[0]
    d = x_multi_layer_list[0].shape[2]
    px = torch.zeros(N, nL, 1, d, dtype=DTYPE, device=DEVICE)
    for i, t in enumerate(x_multi_layer_list):
        px[i, :, 0, :] = t[:, -1, :]  # last token per layer, shape (nL, d)
    return px


# ---------------- training ----------------
def train(arch, regime, ds, train_idx, test_idx, seed, d):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n_layers = ds["x_multi_layer_list"][0].shape[0] if "x_multi_layer_list" in ds else 1
    model = make_probe(arch, d, n_layers).to(DEVICE).to(DTYPE)
    bce = nn.BCEWithLogitsLoss()
    lr = 5e-4 if "attention" in arch else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    train_ix = np.array(train_idx).copy()
    np.random.default_rng(seed).shuffle(train_ix)
    best_loss, best_state, patience = float("inf"), None, 5

    if regime == "batch":
        for epoch in range(50):
            np.random.default_rng(seed + epoch).shuffle(train_ix)
            model.train()
            tl = 0; nb = 0
            for st in range(0, len(train_ix), 32):
                bi = train_ix[st:st+32]
                yb = ds["y"][bi]
                if arch in ("linear", "mlp"):
                    logits = model(ds["x_final"][bi])
                elif arch in ("linear_ml", "mlp_ml", "layer_attn"):
                    px = pad_multi_layer_last([ds["x_multi_layer_list"][i] for i in bi])
                    out = model(None, px, None)
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    # attention probes need 2D: use first layer if multi-layer
                    px_list = []
                    for i in bi:
                        t = ds["x_full_list"][i]
                        if t.dim() == 3 and t.shape[0] > 1:
                            t = t[0]  # first layer only for attention
                        px_list.append(t)
                    px, pm = pad_full(px_list, [ds["mask_list"][i] for i in bi])
                    out = model(None, px, pm)
                    logits = out[0] if isinstance(out, tuple) else out
                loss = bce(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); nb += 1
            ev = evaluate(model, arch, ds, test_idx)
            if ev["loss"] < best_loss - 1e-4:
                best_loss = ev["loss"]; best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 5
            else:
                patience -= 1
                if patience <= 0: break
    else:  # incremental
        model.train()
        for i, idx in enumerate(train_ix):
            yb = ds["y"][idx:idx+1]
            if arch in ("linear", "mlp"):
                logits = model(ds["x_final"][idx:idx+1])
            elif arch in ("linear_ml", "mlp_ml", "layer_attn"):
                px = pad_multi_layer_last([ds["x_multi_layer_list"][idx]])
                out = model(None, px, None)
                logits = out[0] if isinstance(out, tuple) else out
            else:
                t = ds["x_full_list"][idx]
                if t.dim() == 3 and t.shape[0] > 1:
                    t = t[0]
                px, pm = pad_full([t], [ds["mask_list"][idx]])
                out = model(None, px, pm)
                logits = out[0] if isinstance(out, tuple) else out
            loss = bce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return evaluate(model, arch, ds, test_idx), model


def evaluate(model, arch, ds, idx):
    model.eval()
    with torch.no_grad():
        if arch in ("linear", "mlp"):
            logits = model(ds["x_final"][idx])
        elif arch in ("linear_ml", "mlp_ml", "layer_attn"):
            px = pad_multi_layer_last([ds["x_multi_layer_list"][i] for i in idx])
            out = model(None, px, None)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            # attention probes need 2D: use first layer if multi-layer
            px_list = []
            for i in idx:
                t = ds["x_full_list"][i]
                if t.dim() == 3 and t.shape[0] > 1:
                    t = t[0]
                px_list.append(t)
            px, pm = pad_full(px_list, [ds["mask_list"][i] for i in idx])
            out = model(None, px, pm)
            logits = out[0] if isinstance(out, tuple) else out
        y = ds["y"][idx]
        loss = F.binary_cross_entropy_with_logits(logits, y).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.cpu().numpy().astype(int)
        preds = (probs > 0.5).astype(int)
        acc = (preds == y_np).mean()
        try:
            from sklearn.metrics import f1_score, roc_auc_score
            f1 = f1_score(y_np, preds, zero_division=0)
            auc = roc_auc_score(y_np, probs) if len(set(y_np.tolist())) > 1 else float("nan")
        except ImportError:
            f1, auc = float("nan"), float("nan")
    return {"loss": loss, "acc": float(acc), "f1": float(f1), "auc": float(auc)}


def task_specs(manifest):
    cyber = manifest["cyber_samples"]
    for model_key in ["gemma4_31b", "qwen36"]:
        ref = manifest["refusal_samples"][model_key]
        yield (f"refusal_{model_key}", model_key, ref, lambda s: 1.0 if s["is_refusal"] else 0.0)
        for cls in ["prohibited", "high_risk_dual_use", "dual_use", "benign"]:
            pos = [s for s in cyber if s["label"] == cls]
            neg_all = [s for s in cyber if s["label"] != cls]
            other = sorted({s["label"] for s in neg_all})
            n_per = len(pos) // len(other)
            rng = random.Random(42 + hash(cls) % 1000)
            neg = []
            for c in other:
                pool = [s for s in neg_all if s["label"] == c]
                neg += rng.sample(pool, min(n_per, len(pool)))
            samples = pos + neg
            short = {"prohibited": "prohib", "high_risk_dual_use": "hdu",
                    "dual_use": "du", "benign": "ben"}[cls]
            yield (f"cyber_{short}_vs_rest_{model_key}", model_key, samples,
                    (lambda s, _cls=cls: 1.0 if s["label"] == _cls else 0.0))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracts_dir", required=True,
                    help="Directory of per-sample .pt extracts produced by extract_residuals.py")
    ap.add_argument("--manifest", required=True,
                    help="JSON manifest listing samples (with sample_id, label fields)")
    ap.add_argument("--out_dir", default="./probes",
                    help="Where to write probe weights + per-task metrics")
    ap.add_argument("--task", default=None,
                    help="Filter to a single task name (default: run all tasks the manifest defines)")
    return ap.parse_args()


def main():
    args = parse_args()
    extracts_dir = Path(args.extracts_dir)
    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    weights_dir = out_dir / "weights"
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(args.manifest))
    rows = []
    log_path = results_dir / "metrics.jsonl"
    log_f = open(log_path, "w")

    for task_name, model_key, samples, label_fn in task_specs(manifest):
        if args.task and task_name != args.task: continue
        print(f"\n=== {task_name}  ({model_key}, {len(samples)}) ===")
        ds = build_dataset(samples, label_fn, extracts_dir)
        if ds is None:
            print("  no data"); continue
        d = ds["x_final"].shape[1]
        N = len(ds["ids"])
        y_np = ds["y"].cpu().numpy().astype(int)
        pos_idx = np.where(y_np == 1)[0]; neg_idx = np.where(y_np == 0)[0]
        rng = np.random.default_rng(0)
        rng.shuffle(pos_idx); rng.shuffle(neg_idx)
        n_pt = int(len(pos_idx) * 0.3); n_nt = int(len(neg_idx) * 0.3)
        test_idx = np.concatenate([pos_idx[:n_pt], neg_idx[:n_nt]])
        train_idx = np.concatenate([pos_idx[n_pt:], neg_idx[n_nt:]])
        print(f"  d={d} N={N} train={len(train_idx)} test={len(test_idx)} extracts={extracts_dir.name}")

        best_attn_state = None
        best_attn_auc = -1
        best_layer_attn_state = None
        best_layer_attn_auc = -1
        for arch in ARCHS:
            for reg in REGIMES:
                metrics_seeds = []
                for seed in SEEDS:
                    t0 = time.time()
                    metrics, model = train(arch, reg, ds, train_idx, test_idx, seed, d)
                    elapsed = time.time() - t0
                    rec = {"task": task_name, "model_key": model_key, "arch": arch,
                            "regime": reg, "seed": seed, "elapsed_s": round(elapsed, 2),
                            "N_train": int(len(train_idx)), "N_test": int(len(test_idx)),
                            "extracts": extracts_dir.name, **metrics}
                    metrics_seeds.append(metrics)
                    rows.append(rec)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    if arch == "attention" and reg == "batch" and metrics["auc"] > best_attn_auc:
                        best_attn_auc = metrics["auc"]
                        best_attn_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    if arch == "layer_attn" and reg == "batch" and metrics["auc"] > best_layer_attn_auc:
                        best_layer_attn_auc = metrics["auc"]
                        best_layer_attn_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                accs = [m["acc"] for m in metrics_seeds]
                aucs = [m["auc"] for m in metrics_seeds if not math.isnan(m["auc"])]
                print(f"    {arch:13s} | {reg:11s} | acc {np.mean(accs):.3f}±{np.std(accs):.3f} | auc {(np.mean(aucs) if aucs else float('nan')):.3f}±{(np.std(aucs) if aucs else float('nan')):.3f}")
        if best_attn_state is not None:
            torch.save({
                "state": best_attn_state,
                "task": task_name, "model_key": model_key, "arch": "attention",
                "extracts_dir": extracts_dir.name,
                "test_idx": test_idx.tolist(),
                "train_idx": train_idx.tolist(),
                "d_model": d,
                "best_auc_seed_max": best_attn_auc,
            }, str(weights_dir / f"{task_name}_attention.pt"))
        if best_layer_attn_state is not None:
            torch.save({
                "state": best_layer_attn_state,
                "task": task_name, "model_key": model_key, "arch": "layer_attn",
                "extracts_dir": extracts_dir.name,
                "test_idx": test_idx.tolist(),
                "train_idx": train_idx.tolist(),
                "d_model": d,
                "best_auc_seed_max": best_layer_attn_auc,
            }, str(weights_dir / f"{task_name}_layer_attn.pt"))

    log_f.close()
    print(f"\nFull metrics: {log_path}")
    print(f"Probe weights: {weights_dir}")


if __name__ == "__main__":
    main()

