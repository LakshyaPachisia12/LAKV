"""
LAKV Module 1: CalibrationProfiler

Runs ONCE offline on calibration examples. Computes three signals per layer
(attention importance, offset variance, effective rank), produces a joint score,
and assigns each of the 28 layers to Tier 1/2/3. Saves a LayerProfile JSON to
disk that all other modules load.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm


# ─── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class LayerProfile:
    model_name: str
    dataset_name: str
    n_calibration_examples: int
    n_layers: int  # 28 for Qwen2.5-7B

    attention_importance: List[float]          # raw A_l
    offset_variance: List[float]              # raw V_l
    effective_rank: List[float]               # raw R_l

    attention_importance_norm: List[float]     # normalised to [0, 1]
    offset_variance_norm: List[float]
    effective_rank_norm: List[float]

    joint_score: List[float]                  # Score_l
    tier_assignment: List[int]                # 1 / 2 / 3
    estimated_bytes_per_layer: List[float]

    tier_thresholds: dict = field(default_factory=lambda: {"tier1": 0.30, "tier2": 0.70})
    mean_seq_len: float = 0.0

    # ── persistence helpers ──────────────────────────────────────────────
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LayerProfile":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ─── profiler ─────────────────────────────────────────────────────────────────

class CalibrationProfiler:
    """Compute per-layer calibration signals for a Qwen2.5-7B model."""

    SYSTEM_PROMPT = (
        "You are a mathematical reasoning agent. Analyze the problem carefully."
    )

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.N_LAYERS = model.config.num_hidden_layers
        self.N_KV_HEADS = model.config.num_key_value_heads
        self.HEAD_DIM = getattr(model.config, 'head_dim',
                                 model.config.hidden_size // model.config.num_attention_heads)

    # ── public API ────────────────────────────────────────────────────────

    def run_calibration(
        self,
        examples: List[str],
        save_path: str,
        dataset_name: str = "gsm8k",
    ) -> LayerProfile:
        """Run calibration on *examples* and persist the resulting profile."""

        print(f"[CalibrationProfiler] Running calibration on {len(examples)} examples …")

        all_A: List[torch.Tensor] = []   # per-example A_l vectors (shape [28])
        all_V: List[torch.Tensor] = []
        all_R: List[torch.Tensor] = []
        seq_lens: List[int] = []

        for idx, question in enumerate(tqdm(examples, desc="Calibrating")):
            A_l = self._compute_attention_importance(question)
            V_l = self._compute_offset_variance(question)
            R_l, seq_len = self._compute_effective_rank(question)

            all_A.append(A_l)
            all_V.append(V_l)
            all_R.append(R_l)
            seq_lens.append(seq_len)

        # average across examples
        A = torch.stack(all_A).mean(dim=0)   # [28]
        V = torch.stack(all_V).mean(dim=0)
        R = torch.stack(all_R).mean(dim=0)
        mean_seq_len = float(np.mean(seq_lens))

        # Replace any NaN / Inf that slipped through (e.g. degenerate attention
        # in the last layer) with the per-tensor mean so normalisation is clean.
        A = torch.nan_to_num(A, nan=float(A[~A.isnan()].mean()) if A.isnan().any() else 0.0)
        V = torch.nan_to_num(V, nan=0.0)
        R = torch.nan_to_num(R, nan=0.0)

        # normalise to [0, 1]
        A_norm = self._minmax(A)
        V_norm = self._minmax(V)
        R_norm = self._minmax(R)

        # preliminary bit assignment based on R_l_norm
        preliminary_tier = []
        for r in R_norm.tolist():
            preliminary_tier.append(1 if r < 0.33 else 2)

        # estimated compressed bytes per layer
        est_bytes = []
        for pt in preliminary_tier:
            if pt == 1:
                b = self.N_KV_HEADS * mean_seq_len * self.HEAD_DIM * 1  # INT8
            else:
                b = self.N_KV_HEADS * mean_seq_len * self.HEAD_DIM * 0.5  # INT4
            est_bytes.append(b)
        est_bytes_t = torch.tensor(est_bytes, dtype=torch.float32)
        est_bytes_norm = self._minmax(est_bytes_t)

        # joint score
        score = (A_norm * (1.0 - V_norm)) / (est_bytes_norm + 1e-6)
        score = score.tolist()

        # final tier assignment – sort by score descending
        sorted_indices = sorted(range(self.N_LAYERS), key=lambda i: score[i], reverse=True)
        tier1_cutoff = int(math.ceil(self.N_LAYERS * 0.30))
        tier2_cutoff = int(math.ceil(self.N_LAYERS * 0.70))

        tier_assignment = [0] * self.N_LAYERS
        for rank, layer_idx in enumerate(sorted_indices):
            if rank < tier1_cutoff:
                tier_assignment[layer_idx] = 1
            elif rank < tier2_cutoff:
                tier_assignment[layer_idx] = 2
            else:
                tier_assignment[layer_idx] = 3

        profile = LayerProfile(
            model_name=self.model.config._name_or_path if hasattr(self.model, "config") else "unknown",
            dataset_name=dataset_name,
            n_calibration_examples=len(examples),
            n_layers=self.N_LAYERS,
            attention_importance=A.tolist(),
            offset_variance=V.tolist(),
            effective_rank=R.tolist(),
            attention_importance_norm=A_norm.tolist(),
            offset_variance_norm=V_norm.tolist(),
            effective_rank_norm=R_norm.tolist(),
            joint_score=score,
            tier_assignment=tier_assignment,
            estimated_bytes_per_layer=est_bytes,
            tier_thresholds={"tier1": 0.30, "tier2": 0.70},
            mean_seq_len=mean_seq_len,
        )
        profile.save(save_path)
        print(f"[CalibrationProfiler] Profile saved to {save_path}")

        # summary
        n1 = tier_assignment.count(1)
        n2 = tier_assignment.count(2)
        n3 = tier_assignment.count(3)
        print(f"  Tier 1 (INT8, keep): {n1} layers  |  "
              f"Tier 2 (INT4, keep): {n2} layers  |  "
              f"Tier 3 (DROP):       {n3} layers")

        return profile

    # ── Signal 1: Attention Importance ────────────────────────────────────

    def _compute_attention_importance(self, question: str) -> torch.Tensor:
        """A_l = 1 / (mean_entropy + ε) from attention weight matrices.

        Requires the model to be loaded with attn_implementation='eager' so
        that output_attentions=True is respected (sdpa/flash do not return
        attention weight tensors). We also call set_attn_implementation here
        as a belt-and-suspenders guard.
        """
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Ensure eager attention so output_attentions works
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if outputs.attentions is None:
            raise RuntimeError(
                "outputs.attentions is None — load the model with "
                "attn_implementation='eager' in AutoModelForCausalLM.from_pretrained()"
            )

        # outputs.attentions: tuple of (batch, n_heads, seq, seq) per layer
        A_l = torch.zeros(self.N_LAYERS)
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights: (1, n_heads, seq, seq)
            # Cast to float32 first — bfloat16 entropy can overflow to ±inf
            w = attn_weights[0].float().clamp(min=0.0)  # (n_heads, seq, seq)
            # Renormalise rows in case of fp16 rounding drift
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
            log_w = torch.log(w + 1e-9)
            entropy = -(w * log_w).sum(dim=-1)           # (n_heads, seq)
            # Guard: replace any NaN/Inf from degenerate heads with zeros
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            mean_entropy = entropy.mean().item()
            # Clamp to avoid 1/0; negative entropy is impossible, treat as zero
            mean_entropy = max(mean_entropy, 0.0)
            A_l[layer_idx] = 1.0 / (mean_entropy + 1e-6)

        return A_l.cpu()

    # ── Signal 2: Offset Variance ─────────────────────────────────────────

    def _compute_offset_variance(self, question: str) -> torch.Tensor:
        """V_l = ||KV_long - KV_short||_F^2"""
        # Pass 1: question only
        prompt_short = self._build_prompt(question, use_system=False)
        # Pass 2: system_prompt + question
        prompt_long = self._build_prompt(question, use_system=True)

        kv_short = self._get_kv(prompt_short)
        kv_long = self._get_kv(prompt_long)

        V_l = torch.zeros(self.N_LAYERS)
        for layer_idx in range(self.N_LAYERS):
            k_short, v_short = kv_short[layer_idx][0], kv_short[layer_idx][1]
            k_long, v_long = kv_long[layer_idx][0], kv_long[layer_idx][1]

            # Flatten and compute Frobenius norm squared of difference
            # Trim to the shorter sequence length for alignment
            min_seq = min(k_short.shape[2], k_long.shape[2])
            kv_s = torch.cat([k_short[:, :, :min_seq, :], v_short[:, :, :min_seq, :]], dim=-1).flatten().float()
            kv_l = torch.cat([k_long[:, :, :min_seq, :], v_long[:, :, :min_seq, :]], dim=-1).flatten().float()

            V_l[layer_idx] = (kv_l - kv_s).pow(2).sum().item()

        return V_l.cpu()

    # ── Signal 3: Effective Rank ──────────────────────────────────────────

    def _compute_effective_rank(self, question: str):
        """R_l = participation ratio of SVD singular values of K reshaped."""
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]

        pkv = self._get_kv(prompt)

        R_l = torch.zeros(self.N_LAYERS)
        for layer_idx in range(self.N_LAYERS):
            k = pkv[layer_idx][0]  # (1, n_kv_heads, seq_len, head_dim)
            # reshape to (n_kv_heads * seq_len, head_dim)
            matrix = k.squeeze(0).reshape(-1, self.HEAD_DIM).float()
            _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
            participation_ratio = (S.sum() ** 2) / (S.pow(2).sum() + 1e-9)
            R_l[layer_idx] = participation_ratio.item() / self.HEAD_DIM

        return R_l.cpu(), seq_len

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_prompt(self, question: str, use_system: bool = True) -> str:
        if use_system:
            return (
                f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            return (
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _get_kv(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        return self._to_legacy_cache(outputs.past_key_values)

    @staticmethod
    def _to_legacy_cache(pkv):
        """Convert cache objects (e.g. DynamicCache) into ((K,V), ...) tuples."""
        if isinstance(pkv, tuple):
            return pkv

        # Transformers DynamicCache (stable API in 4.36+)
        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()

        # Fallback for cache-like iterables across versions
        if hasattr(pkv, "__iter__"):
            layers = []
            for layer in pkv:
                if isinstance(layer, tuple) and len(layer) >= 2:
                    layers.append((layer[0], layer[1]))
                elif hasattr(layer, "keys") and hasattr(layer, "values"):
                    layers.append((layer.keys, layer.values))
            if layers:
                return tuple(layers)

        raise TypeError(f"Unsupported cache type: {type(pkv)}")

    @staticmethod
    def _minmax(t: torch.Tensor) -> torch.Tensor:
        """NaN-safe min-max normalisation to [0, 1]."""
        # Use nanmin/nanmax so a single NaN layer does not corrupt all values
        t_min = t[~t.isnan()].min() if t.isnan().any() else t.min()
        t_max = t[~t.isnan()].max() if t.isnan().any() else t.max()
        normed = (t - t_min) / (t_max - t_min + 1e-8)
        # Replace any residual NaN (e.g. from constant tensors) with 0
        return torch.nan_to_num(normed, nan=0.0)


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_signals(profile: LayerProfile, save_path: str) -> None:
    """2×2 panel of normalised calibration signals with tier colours."""
    n = profile.n_layers
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"LAKV Layer Profile — {profile.model_name}", fontsize=15, fontweight="bold")

    # Plot 1 – attention importance
    axes[0, 0].bar(x, profile.attention_importance_norm, color="crimson", alpha=0.85)
    axes[0, 0].set_title("Attention Importance  (A_l norm)")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Value")

    # Plot 2 – offset variance
    axes[0, 1].bar(x, profile.offset_variance_norm, color="royalblue", alpha=0.85)
    axes[0, 1].set_title("Offset Variance  (V_l norm)")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Value")

    # Plot 3 – effective rank
    axes[1, 0].bar(x, profile.effective_rank_norm, color="seagreen", alpha=0.85)
    axes[1, 0].set_title("Effective Rank  (R_l norm)")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Value")

    # Plot 4 – joint score with tier colours
    tier_colors = {1: "crimson", 2: "orange", 3: "silver"}
    colors = [tier_colors[t] for t in profile.tier_assignment]
    axes[1, 1].bar(x, profile.joint_score, color=colors, alpha=0.85)
    axes[1, 1].set_title("Joint Score (tier colours: T1=red, T2=orange, T3=grey)")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Score")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_signals] Saved → {save_path}")


def plot_score_scatter(profile: LayerProfile, save_path: str) -> None:
    """Scatter of A_l_norm vs R_l_norm, coloured by joint score."""
    A = np.array(profile.attention_importance_norm)
    R = np.array(profile.effective_rank_norm)
    S = np.array(profile.joint_score)

    r_val, _ = pearsonr(A, R)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(A, R, c=S, cmap="RdYlGn", s=80, edgecolors="k", linewidth=0.5)
    for i in range(len(A)):
        ax.annotate(str(i), (A[i], R[i]), fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Attention Importance (A_l norm)")
    ax.set_ylabel("Effective Rank (R_l norm)")
    ax.set_title(f"Attention Importance vs Effective Rank  (Pearson r = {r_val:.3f})")
    plt.colorbar(sc, label="Joint Score")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_score_scatter] Saved → {save_path}")
