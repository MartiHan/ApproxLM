from typing import Any, Dict

import numpy as np
import pandas as pd


def _to_numpy_attention(x: Any) -> np.ndarray | None:
    if x is None:
        return None

    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        return arr[:, :, :, 0, :]
    raise ValueError(f"Expected attention array with 4D or 5D shape, got {arr.shape}")


def metrics_to_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    if "bleu" in metrics or "rouge_l" in metrics or "perplexity" in metrics or "bertscore_f1" in metrics:
        rows = [{"metric": key, "value": value} for key, value in metrics.items()]
        return pd.DataFrame(rows)

    rows = []
    for key in ["accuracy", "macro avg", "weighted avg"]:
        if key not in metrics:
            continue
        if key == "accuracy":
            rows.append(
                {
                    "label": "accuracy",
                    "precision": metrics[key],
                    "recall": metrics[key],
                    "f1-score": metrics[key],
                    "support": metrics[key],
                }
            )
        else:
            rows.append({"label": key, **metrics[key]})
    return pd.DataFrame(rows)


def experiment_display_label(row: pd.Series) -> str:
    return f"{row['experiment_name']} | {row['created_at']} | {row['experiment_id']}"


def _to_numpy_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def _to_numpy_3d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    return arr


def _index_trace_array(traces: Dict[str, Any], key: str, indices: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(traces[key])
    except ValueError:
        arr = np.asarray(traces[key], dtype=object)
    return arr[indices]


def align_trace_arrays(base_traces: Dict[str, Any], cmp_traces: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if "sample_id" not in base_traces or "sample_id" not in cmp_traces:
        raise RuntimeError("Both experiments must contain sample_id in traces.")

    base_ids = np.asarray(base_traces["sample_id"], dtype=np.int64)
    cmp_ids = np.asarray(cmp_traces["sample_id"], dtype=np.int64)

    base_map = {sample_id: idx for idx, sample_id in enumerate(base_ids.tolist())}
    cmp_map = {sample_id: idx for idx, sample_id in enumerate(cmp_ids.tolist())}
    common_ids = sorted(set(base_map) & set(cmp_map))
    if not common_ids:
        raise RuntimeError("No overlapping sample_id values found between experiments.")

    base_idx = np.asarray([base_map[sample_id] for sample_id in common_ids], dtype=np.int64)
    cmp_idx = np.asarray([cmp_map[sample_id] for sample_id in common_ids], dtype=np.int64)
    out = {
        "sample_id": np.asarray(common_ids, dtype=np.int64),
        "base_idx": base_idx,
        "cmp_idx": cmp_idx,
    }

    if "logits" in base_traces and "logits" in cmp_traces:
        out["base_logits"] = _to_numpy_2d(base_traces["logits"])[base_idx]
        out["cmp_logits"] = _to_numpy_2d(cmp_traces["logits"])[cmp_idx]

    if "cls_by_layer" in base_traces and "cls_by_layer" in cmp_traces:
        out["base_cls_by_layer"] = _to_numpy_3d(base_traces["cls_by_layer"])[base_idx]
        out["cmp_cls_by_layer"] = _to_numpy_3d(cmp_traces["cls_by_layer"])[cmp_idx]

    if "attentions" in base_traces and "attentions" in cmp_traces:
        base_attn = _to_numpy_attention(base_traces["attentions"])
        cmp_attn = _to_numpy_attention(cmp_traces["attentions"])
        if base_attn is not None and cmp_attn is not None and base_attn.shape[1:] == cmp_attn.shape[1:]:
            out["base_attentions"] = base_attn[base_idx]
            out["cmp_attentions"] = cmp_attn[cmp_idx]

    for key in ["generated_token_ids", "topk_token_ids", "hidden_by_layer"]:
        if key in base_traces and key in cmp_traces:
            out[f"base_{key}"] = _index_trace_array(base_traces, key, base_idx)
            out[f"cmp_{key}"] = _index_trace_array(cmp_traces, key, cmp_idx)

    return out


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def kl_divergence_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def cosine_distance_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    sim = dot / np.clip(na * nb, eps, None)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def summarize_vector(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _as_2d_int_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=np.int64)
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)


def _valid_generated_tokens(tokens: Any) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    return arr[arr >= 0]


def _first_divergence(base_tokens: np.ndarray, cmp_tokens: np.ndarray) -> int:
    max_len = max(len(base_tokens), len(cmp_tokens))
    for idx in range(max_len):
        base_val = base_tokens[idx] if idx < len(base_tokens) else None
        cmp_val = cmp_tokens[idx] if idx < len(cmp_tokens) else None
        if base_val != cmp_val:
            return idx
    return max_len


def _compute_sequence_divergence(base_sequences: np.ndarray, cmp_sequences: np.ndarray) -> Dict[str, Any]:
    exact_matches = []
    token_agreements = []
    first_divergences = []
    normalized_first_divergences = []
    length_deltas = []
    max_steps = 0
    per_step_total: list[int] = []
    per_step_mismatch: list[int] = []

    for base_raw, cmp_raw in zip(base_sequences, cmp_sequences):
        base_tokens = _valid_generated_tokens(base_raw)
        cmp_tokens = _valid_generated_tokens(cmp_raw)
        max_len = max(len(base_tokens), len(cmp_tokens))
        min_len = min(len(base_tokens), len(cmp_tokens))
        max_steps = max(max_steps, max_len)
        exact_matches.append(float(np.array_equal(base_tokens, cmp_tokens)))
        length_deltas.append(float(len(cmp_tokens) - len(base_tokens)))

        if max_len == 0:
            token_agreements.append(1.0)
        else:
            matches = int(np.sum(base_tokens[:min_len] == cmp_tokens[:min_len]))
            token_agreements.append(matches / max_len)

        first_idx = _first_divergence(base_tokens, cmp_tokens)
        first_divergences.append(float(first_idx))
        normalized_first_divergences.append(float(first_idx / max(max_len, 1)))

        while len(per_step_total) < max_len:
            per_step_total.append(0)
            per_step_mismatch.append(0)
        for step_idx in range(max_len):
            base_val = base_tokens[step_idx] if step_idx < len(base_tokens) else None
            cmp_val = cmp_tokens[step_idx] if step_idx < len(cmp_tokens) else None
            per_step_total[step_idx] += 1
            if base_val != cmp_val:
                per_step_mismatch[step_idx] += 1

    per_step_divergence = [
        mismatch / total if total else 0.0
        for mismatch, total in zip(per_step_mismatch, per_step_total)
    ]
    cumulative_mismatch = np.cumsum(np.asarray(per_step_mismatch, dtype=np.float64))
    cumulative_total = np.cumsum(np.asarray(per_step_total, dtype=np.float64))
    cumulative_divergence = (cumulative_mismatch / np.clip(cumulative_total, 1.0, None)).tolist()

    return {
        "sequence_divergence": {
            "exact_sequence_match_rate": float(np.mean(exact_matches)) if exact_matches else float("nan"),
            **summarize_vector(np.asarray(token_agreements), "token_agreement"),
            **summarize_vector(np.asarray(first_divergences), "first_divergence_index"),
            **summarize_vector(np.asarray(normalized_first_divergences), "normalized_first_divergence"),
            **summarize_vector(np.asarray(length_deltas), "generated_length_delta"),
        },
        "cumulative_generation_drift": {
            "per_step_divergence_rate": per_step_divergence,
            "cumulative_divergence_rate": cumulative_divergence,
            "max_generation_steps": int(max_steps),
        },
    }


def _compute_topk_token_agreement(
    base_topk_values: np.ndarray,
    cmp_topk_values: np.ndarray,
    base_sequences: np.ndarray | None = None,
    cmp_sequences: np.ndarray | None = None,
) -> Dict[str, Any]:
    top1_matches = []
    jaccards = []
    cmp_token_in_base_topk = []
    base_token_in_cmp_topk = []
    per_step_matches: list[list[float]] = []

    for sample_idx, (base_raw, cmp_raw) in enumerate(zip(base_topk_values, cmp_topk_values)):
        base_topk = _as_2d_int_array(base_raw)
        cmp_topk = _as_2d_int_array(cmp_raw)
        steps = min(base_topk.shape[0], cmp_topk.shape[0])
        if steps == 0:
            continue

        base_tokens = _valid_generated_tokens(base_sequences[sample_idx]) if base_sequences is not None else None
        cmp_tokens = _valid_generated_tokens(cmp_sequences[sample_idx]) if cmp_sequences is not None else None

        while len(per_step_matches) < steps:
            per_step_matches.append([])
        for step_idx in range(steps):
            base_ids = [int(token_id) for token_id in base_topk[step_idx] if int(token_id) >= 0]
            cmp_ids = [int(token_id) for token_id in cmp_topk[step_idx] if int(token_id) >= 0]
            if not base_ids or not cmp_ids:
                continue
            top1_match = float(base_ids[0] == cmp_ids[0])
            top1_matches.append(top1_match)
            per_step_matches[step_idx].append(top1_match)
            base_set = set(base_ids)
            cmp_set = set(cmp_ids)
            jaccards.append(len(base_set & cmp_set) / max(len(base_set | cmp_set), 1))
            if cmp_tokens is not None and step_idx < len(cmp_tokens):
                cmp_token_in_base_topk.append(float(int(cmp_tokens[step_idx]) in base_set))
            if base_tokens is not None and step_idx < len(base_tokens):
                base_token_in_cmp_topk.append(float(int(base_tokens[step_idx]) in cmp_set))

    stats = {
        **summarize_vector(np.asarray(top1_matches), "top1_agreement"),
        **summarize_vector(np.asarray(jaccards), "topk_jaccard"),
        "per_step_top1_agreement": [
            float(np.mean(values)) if values else float("nan")
            for values in per_step_matches
        ],
    }
    if cmp_token_in_base_topk:
        stats["cmp_generated_token_in_base_topk_rate"] = float(np.mean(cmp_token_in_base_topk))
    if base_token_in_cmp_topk:
        stats["base_generated_token_in_cmp_topk_rate"] = float(np.mean(base_token_in_cmp_topk))
    return stats


def _compute_hidden_state_drift(base_hidden_values: np.ndarray, cmp_hidden_values: np.ndarray) -> Dict[str, Any]:
    l2_values = []
    rel_l2_values = []
    cosine_values = []
    final_l2_values = []
    final_rel_l2_values = []
    final_cosine_values = []
    per_layer_rel_l2: list[list[float]] = []
    per_layer_cosine: list[list[float]] = []
    per_step_rel_l2: list[list[float]] = []

    for base_raw, cmp_raw in zip(base_hidden_values, cmp_hidden_values):
        base_hidden = np.asarray(base_raw, dtype=np.float32)
        cmp_hidden = np.asarray(cmp_raw, dtype=np.float32)
        if base_hidden.ndim != 3 or cmp_hidden.ndim != 3:
            continue
        steps = min(base_hidden.shape[0], cmp_hidden.shape[0])
        layers = min(base_hidden.shape[1], cmp_hidden.shape[1])
        hidden = min(base_hidden.shape[2], cmp_hidden.shape[2])
        if steps == 0 or layers == 0 or hidden == 0:
            continue

        base_aligned = base_hidden[:steps, :layers, :hidden]
        cmp_aligned = cmp_hidden[:steps, :layers, :hidden]
        l2 = np.linalg.norm(base_aligned - cmp_aligned, axis=-1)
        rel_l2 = l2 / np.clip(np.linalg.norm(base_aligned, axis=-1), 1e-8, None)
        cosine = cosine_distance_np(base_aligned, cmp_aligned)
        l2_values.extend(l2.reshape(-1).tolist())
        rel_l2_values.extend(rel_l2.reshape(-1).tolist())
        cosine_values.extend(cosine.reshape(-1).tolist())
        final_l2_values.extend(l2[-1].reshape(-1).tolist())
        final_rel_l2_values.extend(rel_l2[-1].reshape(-1).tolist())
        final_cosine_values.extend(cosine[-1].reshape(-1).tolist())

        while len(per_layer_rel_l2) < layers:
            per_layer_rel_l2.append([])
            per_layer_cosine.append([])
        while len(per_step_rel_l2) < steps:
            per_step_rel_l2.append([])
        for layer_idx in range(layers):
            per_layer_rel_l2[layer_idx].extend(rel_l2[:, layer_idx].tolist())
            per_layer_cosine[layer_idx].extend(cosine[:, layer_idx].tolist())
        for step_idx in range(steps):
            per_step_rel_l2[step_idx].extend(rel_l2[step_idx].tolist())

    return {
        **summarize_vector(np.asarray(l2_values), "l2"),
        **summarize_vector(np.asarray(rel_l2_values), "rel_l2"),
        **summarize_vector(np.asarray(cosine_values), "cosine"),
        **summarize_vector(np.asarray(final_l2_values), "final_step_l2"),
        **summarize_vector(np.asarray(final_rel_l2_values), "final_step_rel_l2"),
        **summarize_vector(np.asarray(final_cosine_values), "final_step_cosine"),
        "per_layer_mean_rel_l2": [float(np.mean(values)) if values else float("nan") for values in per_layer_rel_l2],
        "per_layer_mean_cosine": [float(np.mean(values)) if values else float("nan") for values in per_layer_cosine],
        "per_step_mean_rel_l2": [float(np.mean(values)) if values else float("nan") for values in per_step_rel_l2],
    }


def compute_drift_statistics(base_exp: Dict[str, Any], cmp_exp: Dict[str, Any]) -> Dict[str, Any]:
    base_traces = base_exp.get("traces", {})
    cmp_traces = cmp_exp.get("traces", {})
    if not base_traces or not cmp_traces:
        raise RuntimeError(
            "Both experiments must have saved trace artifacts. "
            "Rerun older/no-trace experiments before computing drift."
        )

    aligned = align_trace_arrays(base_traces, cmp_traces)
    stats: Dict[str, Any] = {"num_aligned_samples": int(len(aligned["sample_id"]))}

    if "base_logits" in aligned and "cmp_logits" in aligned:
        base_logits = aligned["base_logits"]
        cmp_logits = aligned["cmp_logits"]
        logit_l2 = np.linalg.norm(base_logits - cmp_logits, axis=-1)
        logit_kl = kl_divergence_np(softmax_np(base_logits, axis=-1), softmax_np(cmp_logits, axis=-1))
        pred_flip = (np.argmax(base_logits, axis=-1) != np.argmax(cmp_logits, axis=-1)).astype(np.float32)
        stats["logit_drift"] = {
            **summarize_vector(logit_l2, "l2"),
            **summarize_vector(logit_kl, "kl"),
            "prediction_flip_rate": float(np.mean(pred_flip)),
            "kl_values": logit_kl.tolist(),
            "l2_values": logit_l2.tolist(),
        }

    if "base_attentions" in aligned and "cmp_attentions" in aligned:
        attn_abs = np.mean(np.abs(aligned["base_attentions"] - aligned["cmp_attentions"]), axis=-1)
        per_layer_head_mean = np.mean(attn_abs, axis=0)
        stats["attention_drift"] = {
            "per_layer_head_mean_abs": per_layer_head_mean.tolist(),
            "per_layer_mean_abs": np.mean(per_layer_head_mean, axis=-1).tolist(),
            "global_mean_abs": float(np.mean(attn_abs)),
            "global_std_abs": float(np.std(attn_abs)),
        }

    if "base_cls_by_layer" in aligned and "cmp_cls_by_layer" in aligned:
        base_cls = aligned["base_cls_by_layer"]
        cmp_cls = aligned["cmp_cls_by_layer"]
        base_final = base_cls[:, -1, :]
        cmp_final = cmp_cls[:, -1, :]
        cls_l2 = np.linalg.norm(base_final - cmp_final, axis=-1)
        cls_rel_l2 = cls_l2 / np.clip(np.linalg.norm(base_final, axis=-1), 1e-8, None)
        cls_cos = cosine_distance_np(base_final, cmp_final)
        cls_stats = {
            **summarize_vector(cls_l2, "l2"),
            **summarize_vector(cls_rel_l2, "rel_l2"),
            **summarize_vector(cls_cos, "cosine"),
        }

        per_layer_rel_l2 = []
        per_layer_cos = []
        for layer_idx in range(base_cls.shape[1]):
            base_layer = base_cls[:, layer_idx, :]
            cmp_layer = cmp_cls[:, layer_idx, :]
            layer_l2 = np.linalg.norm(base_layer - cmp_layer, axis=-1)
            per_layer_rel_l2.append(
                float(np.mean(layer_l2 / np.clip(np.linalg.norm(base_layer, axis=-1), 1e-8, None)))
            )
            per_layer_cos.append(float(np.mean(cosine_distance_np(base_layer, cmp_layer))))

        cls_stats["per_layer_mean_rel_l2"] = per_layer_rel_l2
        cls_stats["per_layer_mean_cosine"] = per_layer_cos
        stats["cls_embedding_drift"] = cls_stats

    if "base_generated_token_ids" in aligned and "cmp_generated_token_ids" in aligned:
        sequence_stats = _compute_sequence_divergence(
            aligned["base_generated_token_ids"],
            aligned["cmp_generated_token_ids"],
        )
        stats.update(sequence_stats)

    if "base_topk_token_ids" in aligned and "cmp_topk_token_ids" in aligned:
        stats["topk_token_agreement"] = _compute_topk_token_agreement(
            aligned["base_topk_token_ids"],
            aligned["cmp_topk_token_ids"],
            aligned.get("base_generated_token_ids"),
            aligned.get("cmp_generated_token_ids"),
        )

    if "base_hidden_by_layer" in aligned and "cmp_hidden_by_layer" in aligned:
        stats["hidden_state_drift"] = _compute_hidden_state_drift(
            aligned["base_hidden_by_layer"],
            aligned["cmp_hidden_by_layer"],
        )

    return stats
