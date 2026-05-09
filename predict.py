from __future__ import annotations
from pathlib import Path
import json, math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Repository root finding
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "datasets").exists() and (current / "extracts").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate repo root")


_ROOT = find_repo_root(Path(__file__))
_DATA = _ROOT / "datasets"
_EXTRACTS = _ROOT / "extracts"


# ---------------------------------------------------------------------------
# Probe architectures (must match train_probe.py)
# ---------------------------------------------------------------------------

class AttentionProbe(nn.Module):
    """Single learned-query attention over tokens."""
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)

    def forward(self, x_final, x_full, mask):
        # x_full: (B, N, d) — all tokens
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)
        return self.head(pooled).squeeze(-1), alpha


class LinearProbe(nn.Module):
    """Linear probe over multiple layers: concatenate layer outputs then project."""
    def __init__(self, d, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.fc = nn.Linear(d * n_layers, 1)

    def forward(self, x_final, x_full, mask):
        # x_full: (B, n_layers, d)
        B, nL, d = x_full.shape
        x = x_full.reshape(B, nL * d)
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    """MLP probe over multiple layers with hidden layer."""
    def __init__(self, d, n_layers, hidden=256, drop=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.net = nn.Sequential(
            nn.Linear(d * n_layers, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))

    def forward(self, x_final, x_full, mask):
        B, nL, d = x_full.shape
        x = x_full.reshape(B, nL * d)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Residual loading
# ---------------------------------------------------------------------------

def _load_extract(sample_id: str, model_key: str) -> dict:
    """Load a single .pt extract saved by extract_residuals."""
    path = _EXTRACTS / model_key / f"{sample_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No extract: {path}")
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Load trained probe weights
# ---------------------------------------------------------------------------

def _detect_architecture(state_dict: dict, d_model: int) -> tuple[str, int]:
    """Detect probe architecture from state dict keys."""
    keys = set(state_dict.keys())
    if "q" in keys and "head.weight" in keys and len(keys) == 3:
        # AttentionProbe: q, head.weight, head.bias
        return "attention", 1
    elif "fc.weight" in keys and state_dict["fc.weight"].shape[0] == 1:
        # LinearProbe: fc.weight, fc.bias
        n_layers = state_dict["fc.weight"].shape[1] // d_model
        return "linear_ml", n_layers
    elif "net.0.weight" in keys:
        # MLPProbe: net.0.weight (Linear), net.2.weight (Linear), ...
        n_layers = state_dict["net.0.weight"].shape[0] // d_model
        return "mlp_ml", n_layers
    else:
        raise ValueError(f"Unknown architecture: {keys}")


def load_probe_from_checkpoint(probe_path: Path) -> tuple[nn.Module, dict]:
    """Load a trained probe from checkpoint."""
    ckpt = torch.load(probe_path, weights_only=False)
    task = ckpt["task"]
    d_model = ckpt["d_model"]

    state_dict = ckpt["state"]
    arch, n_layers = _detect_architecture(state_dict, d_model)

    # Reconstruct the probe
    if arch == "attention":
        probe = AttentionProbe(d_model)
    elif arch == "linear_ml":
        probe = LinearProbe(d_model, n_layers)
    elif arch == "mlp_ml":
        probe = MLPProbe(d_model, n_layers)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    probe.load_state_dict(state_dict)
    probe.eval()

    return probe, {"task": task, "d_model": d_model, "n_layers": n_layers, "arch": arch}


def load_predictor() -> dict[str, Any]:
    """
    Load all trained probes and return an opaque dict that predict() can use.
    """
    probes_dir = _ROOT / "probes" / "weights"

    tasks = [
        "refusal_gemma4_31b",
        "refusal_qwen36",
    ]

    predictor = {"probes": {}, "meta": {}}

    for task_name in tasks:
        probe_path = probes_dir / f"{task_name}_attention.pt"
        if not probe_path.exists():
            print(f"  [{task_name}] WARNING: no checkpoint at {probe_path}")
            continue

        try:
            probe, meta = load_probe_from_checkpoint(probe_path)
            predictor["probes"][task_name] = probe
            predictor["meta"][task_name] = meta
            print(f"  [{task_name}] loaded arch={meta['arch']} d={meta['d_model']} n_layers={meta['n_layers']}")
        except Exception as e:
            print(f"  [{task_name}] WARNING: {e}")

    print(f"  [load_predictor] done, {len(predictor['probes'])}/2 probes ready")
    return predictor


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(
    predictor: dict,
    residuals: np.ndarray,
    attention_mask: np.ndarray | None = None,
    task: str | None = None,
) -> float:
    """
    Score one test sample using trained probe.

    residuals:     (n_selected_layers, n_tokens, d_model) — e.g., (3, N, 5376)
    attention_mask: (n_tokens,) bool, or None
    task:          which task to score; if None, uses whichever is available
    Returns:       float in [0, 1]
    """
    res = torch.from_numpy(residuals.astype(np.float32))

    if attention_mask is not None:
        mask = torch.from_numpy(attention_mask.astype(bool))
    else:
        mask = torch.ones(res.shape[1], dtype=torch.bool)

    # Get last real token index
    last_idx = int(mask.nonzero().max().item())

    probes = predictor["probes"]

    # If specific task requested
    if task and task in probes:
        probe = probes[task]
        meta = predictor["meta"][task]
        arch = meta["arch"]

        with torch.no_grad():
            if arch == "attention":
                # AttentionProbe needs (B, N, d) all tokens
                x = res.mean(dim=0)  # (N, d)
                x = x.unsqueeze(0)   # (1, N, d)
                logit, alpha = probe(None, x, mask.unsqueeze(0))
            elif arch in ("linear_ml", "mlp_ml"):
                # Multi-layer probes: (B, n_layers, d) last token per layer
                layer_vectors = res[:, last_idx, :]  # (n_layers, d)
                x = layer_vectors.unsqueeze(0)       # (1, n_layers, d)
                out = probe(None, x, None)
                logit = out[0] if isinstance(out, tuple) else out
            else:
                return 0.5

            prob = torch.sigmoid(logit).item()
        return float(prob)

    # Fallback: try all probes, return max
    best_prob = 0.5
    for task_name, probe in probes.items():
        meta = predictor["meta"][task_name]
        arch = meta["arch"]

        with torch.no_grad():
            if arch == "attention":
                x = res.mean(dim=0).unsqueeze(0)
                logit, _ = probe(None, x, mask.unsqueeze(0))
            elif arch in ("linear_ml", "mlp_ml"):
                layer_vectors = res[:, last_idx, :]
                x = layer_vectors.unsqueeze(0)
                out = probe(None, x, None)
                logit = out[0] if isinstance(out, tuple) else out
            else:
                continue

            prob = torch.sigmoid(logit).item()
            if prob > best_prob:
                best_prob = prob

    return float(best_prob)


# ---------------------------------------------------------------------------
# Aliases for harness compatibility
# ---------------------------------------------------------------------------

def load_predictor_legacy() -> dict[str, Any]:
    return load_predictor()