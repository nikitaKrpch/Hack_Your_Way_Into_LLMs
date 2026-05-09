from __future__ import annotations
from pathlib import Path
import json, math
from typing import Any

import numpy as np
import torch




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
# Residual I/O
# ---------------------------------------------------------------------------

def _load_extract(sample_id: str, model_key: str) -> dict:
    """Load a single .pt extract saved by extract_residuals."""
    path = _EXTRACTS / model_key / f"{sample_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No extract: {path}")
    return torch.load(path, weights_only=False)


def _last_vec(extract: dict) -> np.ndarray:
    """(n_layers, n_tokens, d) → float32 vector of the last real token."""
    res   = extract["residuals"]                   # fp16, (n_layers, N, d)
    mask  = extract["attention_mask"]              # bool (N,)
    last  = int(mask.nonzero().max().item())
    # last selected layer (index -1 == "late" or "middle" depending on extract config)
    v = res[-1, last].half().float().numpy()      # (d,)
    return v


# ---------------------------------------------------------------------------
# Probe split helpers (mirrors score_probes.py logic)
# ---------------------------------------------------------------------------

def _cyber_split(category: str, probe_id: int) -> int | None:
    """1 = positive, 0 = negative, None = not in this probe's set."""
    pos = {1: "dual_use", 2: "high_risk_dual_use", 3: "prohibited"}[probe_id]
    neg = {
        1: {"benign"},
        2: {"dual_use", "benign"},
        3: {"high_risk_dual_use", "dual_use", "benign"},
    }[probe_id]
    if category == pos:        return 1
    if category in neg:        return 0
    return None


def _build_centroids(task_key: str) -> dict[str, np.ndarray]:
    """Compute unit-normalised mean vectors for pos and neg classes."""
    if task_key.startswith("cyber"):
        probe_id = int(task_key.split("_probe")[1])
        rows = [json.loads(l) for l in open(_DATA / "cyber_probes" / "train.jsonl")]
        pos_vs, neg_vs = [], []
        for r in rows:
            lbl = _cyber_split(r["category"], probe_id)
            if lbl is None:
                continue
            for mk in ["gemma4_31b", "qwen36"]:
                try:
                    ex = _load_extract(r["sample_id"], mk)
                except FileNotFoundError:
                    continue
                v = _last_vec(ex)
                (pos_vs if lbl == 1 else neg_vs).append(v)
                break
    elif task_key.startswith("refusal"):
        model_key = task_key.split("_", 1)[1]   # "gemma4_31b" or "qwen36"
        rows = [json.loads(l) for l in open(
            _DATA / "refusal_probes" / model_key / "attacks_full.jsonl")]
        pos_vs, neg_vs = [], []
        for r in rows:
            try:
                ex = _load_extract(r["sample_id"], model_key)
            except FileNotFoundError:
                continue
            v = _last_vec(ex)
            (pos_vs if r.get("is_refusal") else neg_vs).append(v)
    else:
        raise ValueError(f"Unknown task {task_key!r}")

    def _mean_unit(vs: list[np.ndarray]) -> np.ndarray:
        if not vs:
            raise ValueError(f"No vectors for task {task_key}")
        mu = np.mean(vs, axis=0)
        n  = np.linalg.norm(mu)
        if n < 1e-9:
            raise ValueError(f"Zero centroid norm for task {task_key}")
        return mu / n

    return {"pos": _mean_unit(pos_vs), "neg": _mean_unit(neg_vs)}


# ---------------------------------------------------------------------------
# load_predictor / load_probe — both exposed so any harness works
# ---------------------------------------------------------------------------

def load_predictor() -> dict[str, Any]:
    return load_probe()


def load_probe() -> dict[str, Any]:
    """
    Pre-compute pos/neg centroids for all 5 scoring tasks and return
    an opaque dict that predict() can use.
    """
    tasks = [
        "cyber_probe1", "cyber_probe2", "cyber_probe3",
        "refusal_gemma4_31b", "refusal_qwen36",
    ]
    centroids = {}
    for t in tasks:
        try:
            centroids[t] = _build_centroids(t)
            n = np.linalg.norm(centroids[t]["pos"])
            print(f"  [{t}] centroid |pos|={n:.4f}", flush=True)
        except Exception as e:
            print(f"  [{t}] WARNING: {e}")
            centroids[t] = None
    print(f"  [load_probe] done, {sum(v is not None for v in centroids.values())}/5 tasks ready")
    return {"centroids": centroids}


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(
    probe: dict,
    residuals: np.ndarray,
    attention_mask: np.ndarray | None = None,
) -> float:
    """
    Score one test sample by cosine similarity to the positive centroid.

    residuals:     (n_selected_layers, n_tokens, d_model) fp16 from extract_residuals
    attention_mask: (n_tokens,) bool, or None
    Returns: float in [0, 1]
    """
    # Extract last-token vector from last selected layer
    if residuals.ndim == 3:
        if attention_mask is not None:
            last_idx = int(np.asarray(attention_mask).nonzero()[0].max())
        else:
            last_idx = residuals.shape[1] - 1
        vec = residuals[-1, last_idx].astype(np.float32)
    elif residuals.ndim == 2:          # fallback: (n_tokens, d)
        vec = residuals[-1].astype(np.float32)
    else:
        return 0.5

    vec_t  = torch.from_numpy(vec).float()
    vec_t  = vec_t / (torch.norm(vec_t) + 1e-9)

    centroids = probe["centroids"]
    best_score = -1.0

    for task_key, c in centroids.items():
        if c is None:
            continue
        c_pos = torch.from_numpy(c["pos"]).float()
        cos   = float(torch.dot(vec_t, c_pos).item())
        prob  = (cos + 1.0) / 2.0           # cosine ∈ [-1,1] → p ∈ [0,1]
        if prob > best_score:
            best_score = prob

    return float(best_score) if best_score >= 0 else 0.5