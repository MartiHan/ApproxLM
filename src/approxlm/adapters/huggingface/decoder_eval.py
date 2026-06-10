from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-GPTQ-Int8"
DEFAULT_DATASET = "tatsu-lab/alpaca"
DEFAULT_WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT2_ALIASES = {"wiki2", "wikitext2", "mindchain/wikitext2"}
DEFAULT_BERTSCORE_MODEL = "bert-base-uncased"
ProgressCallback = Callable[[float, str], None] | None
QUALITATIVE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
QUALITATIVE_TOKEN_BOUNDARY_MARKERS = ("Ġ", "▁")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen decoder-only model on Alpaca with BLEU, ROUGE-L, perplexity, and BERTScore."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Causal LM checkpoint.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset config/subset.")
    parser.add_argument("--dataset-format", default="auto", choices=["auto", "alpaca", "wikitext"], help="How to convert rows into prompts and references.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of examples to evaluate. Use 0 with --all-samples for full split.")
    parser.add_argument("--all-samples", action="store_true", help="Evaluate the full selected split.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size for generation and perplexity.")
    parser.add_argument("--perplexity-batch-size", type=int, default=None, help="Optional micro-batch size for reference perplexity scoring.")
    parser.add_argument("--perplexity-stride", type=int, default=None, help="Optional token stride for WikiText corpus perplexity. Defaults to the corpus length, matching the provided reference code.")
    parser.add_argument("--wikitext-token-limit", type=int, default=None, help="Maximum number of tokenized WikiText split tokens to score. Use 0 or omit for the full split.")
    parser.add_argument("--max-input-length", type=int, default=1024, help="Tokenizer truncation length for prompts.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens per example.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cpu.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to model/tokenizer loading.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Sampling top-k.")
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL, help="Encoder model used for BERTScore.")
    parser.add_argument("--output-json", default=None, help="Optional path to write metrics and per-sample outputs as JSON.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    user_text = instruction.strip()
    if input_text.strip():
        user_text = f"{user_text}\n\nInput:\n{input_text.strip()}"
    return user_text


def load_generation_model(model_name: str, trust_remote_code: bool, device: torch.device):
    gptq_model = "gptq" in model_name.lower()
    tokenizer_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}

    if not gptq_model:
        tokenizer_kwargs["trust_remote_code"] = trust_remote_code
        tokenizer_kwargs["use_fast"] = True
        model_kwargs["trust_remote_code"] = trust_remote_code
        model_kwargs.update(generation_model_load_kwargs(model_name))
    else:
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ImportError as exc:
        raise ImportError(
            f"{exc} | python={sys.executable} | model={model_name} | gptq_mode={gptq_model}"
        ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()
    print(model)
    return tokenizer, model


def generation_model_load_kwargs(model_name: str) -> Dict[str, Any]:
    if "gptq" not in model_name.lower():
        return {}
    return {
        "torch_dtype": "auto",
        "device_map": "auto",
    }


def get_model_input_device(model) -> torch.device:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device
    return next(model.parameters()).device


def load_bertscore_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, instruction: str, input_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": format_alpaca_prompt(instruction, input_text)},
    ]
    return tokenizer.apply_chat_template(messages, enable_thinking=False, tokenize=False, add_generation_prompt=True)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    model_device = get_model_input_device(model)
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_responses_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
) -> List[str]:
    if not prompts:
        return []

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    model_device = get_model_input_device(model)
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    prompt_width = input_ids.shape[1]
    return [
        tokenizer.decode(output_ids[row_idx, prompt_width:], skip_special_tokens=True).strip()
        for row_idx in range(output_ids.shape[0])
    ]


def generate_responses_batched_with_traces(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    trace_top_k: int = 10,
) -> tuple[List[str], Dict[str, Any]]:
    if not prompts:
        return [], {}

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    model_device = get_model_input_device(model)
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        generation_out = model.generate(**generation_kwargs)

    output_ids = generation_out.sequences
    prompt_width = input_ids.shape[1]
    generated_ids = output_ids[:, prompt_width:]
    predictions = [
        tokenizer.decode(generated_ids[row_idx], skip_special_tokens=True).strip()
        for row_idx in range(generated_ids.shape[0])
    ]

    scores = tuple(getattr(generation_out, "scores", ()) or ())
    if scores:
        k = min(int(trace_top_k), scores[0].shape[-1])
        topk_token_ids = torch.stack([torch.topk(step_scores.detach(), k=k, dim=-1).indices for step_scores in scores], dim=1)
    else:
        topk_token_ids = torch.empty((generated_ids.shape[0], 0, 0), dtype=torch.long, device=generated_ids.device)

    hidden_steps = tuple(getattr(generation_out, "hidden_states", ()) or ())
    hidden_by_layer = None
    if hidden_steps:
        step_vectors = []
        for step_hidden in hidden_steps:
            if step_hidden is None or len(step_hidden) <= 1:
                continue
            layer_vectors = [
                layer_state[:, -1, :].detach().to(torch.float32).cpu()
                for layer_state in step_hidden[1:]
                if layer_state is not None
            ]
            if layer_vectors:
                step_vectors.append(torch.stack(layer_vectors, dim=1))
        if step_vectors:
            hidden_by_layer = torch.stack(step_vectors, dim=1).numpy()

    traces = {
        "prompt_input_ids": input_ids.detach().cpu().numpy(),
        "prompt_attention_mask": attention_mask.detach().cpu().numpy(),
        "generated_token_ids": generated_ids.detach().cpu().numpy(),
        "topk_token_ids": topk_token_ids.detach().cpu().numpy(),
    }
    if hidden_by_layer is not None:
        traces["hidden_by_layer"] = hidden_by_layer
    return predictions, traces


def normalize_qualitative_token(token: str) -> str:
    normalized = token.strip()
    while normalized.startswith(QUALITATIVE_TOKEN_BOUNDARY_MARKERS):
        normalized = normalized[1:].strip()
    return normalized


def evaluate_qualitative_prompt(
    model,
    tokenizer,
    prompt: str,
    ground_truth: str,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_token_count: int = 20,
) -> Dict[str, Any]:
    model_device = get_model_input_device(model)
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    prompt_input_ids = prompt_enc["input_ids"].to(model_device)
    prompt_attention_mask = prompt_enc["attention_mask"].to(model_device)

    generation_kwargs = {
        "input_ids": prompt_input_ids,
        "attention_mask": prompt_attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k

    try:
        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    except Exception:
        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                use_cache=False,
                return_dict=True,
            )

    with torch.no_grad():
        generated_ids = model.generate(**generation_kwargs)

    next_token_logits = prompt_outputs.logits[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    candidate_count = min(max(int(top_token_count) * 20, int(top_token_count)), int(next_token_probs.shape[-1]))
    top_probs, top_ids = torch.topk(next_token_probs, k=candidate_count)
    top_tokens = []
    for token_id, prob in zip(top_ids.detach().cpu(), top_probs.detach().cpu()):
        decoded_token = tokenizer.decode([int(token_id.item())])
        normalized_token = normalize_qualitative_token(decoded_token)
        if not normalized_token or QUALITATIVE_TOKEN_PATTERN.fullmatch(normalized_token) is None:
            continue
        top_tokens.append(
            {
                "rank": len(top_tokens) + 1,
                "token_id": int(token_id.item()),
                "token": normalized_token,
                "probability": float(prob.item()),
            }
        )
        if len(top_tokens) >= int(top_token_count):
            break

    generated_text = tokenizer.decode(
        generated_ids[0, prompt_input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    attention_contributions: List[Dict[str, Any]] = []
    attentions = getattr(prompt_outputs, "attentions", None)
    if attentions:
        last_attention = attentions[-1]
        if last_attention is not None and last_attention.ndim == 4:
            final_token_attention = last_attention[0, :, -1, :].detach().to(torch.float32).mean(dim=0)
            prompt_token_ids = prompt_input_ids[0].detach().cpu().tolist()
            prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_token_ids)
            attention_values = final_token_attention.detach().cpu().tolist()
            total_attention = sum(float(value) for value in attention_values) or 1.0
            for position, (token_id, raw_token, attention_value) in enumerate(
                zip(prompt_token_ids, prompt_tokens, attention_values),
                start=1,
            ):
                decoded_token = tokenizer.decode([int(token_id)])
                normalized_token = normalize_qualitative_token(decoded_token) or str(raw_token)
                attention_contributions.append(
                    {
                        "position": position,
                        "token_id": int(token_id),
                        "token": normalized_token,
                        "raw_token": str(raw_token),
                        "attention": float(attention_value),
                        "attention_fraction": float(attention_value) / total_attention,
                    }
                )
            attention_contributions.sort(key=lambda item: item["attention"], reverse=True)

    ground_truth_ids = tokenizer(ground_truth, add_special_tokens=False)["input_ids"]
    token_probabilities: List[Dict[str, Any]] = []
    mean_ground_truth_nll = float("nan")
    ground_truth_perplexity = float("nan")
    if ground_truth_ids:
        combined_ids = prompt_input_ids[0].detach().cpu().tolist() + ground_truth_ids
        if len(combined_ids) > max_input_length + max_new_tokens:
            combined_ids = combined_ids[: max_input_length + max_new_tokens]
        combined = torch.tensor([combined_ids], device=model_device, dtype=torch.long)
        with torch.no_grad():
            combined_logits = model(
                input_ids=combined,
                attention_mask=torch.ones_like(combined),
                use_cache=False,
                return_dict=True,
            ).logits

        prompt_len = int(prompt_input_ids.shape[1])
        nll_values: List[float] = []
        for offset, token_id in enumerate(combined_ids[prompt_len:]):
            logit_index = prompt_len + offset - 1
            if logit_index < 0 or logit_index >= combined_logits.shape[1]:
                continue
            probs = torch.softmax(combined_logits[0, logit_index, :], dim=-1)
            prob = float(probs[token_id].detach().cpu().item())
            nll = -math.log(max(prob, 1e-12))
            nll_values.append(nll)
            token_probabilities.append(
                {
                    "position": offset + 1,
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "probability": prob,
                    "surprisal": nll,
                }
            )
        if nll_values:
            mean_ground_truth_nll = float(sum(nll_values) / len(nll_values))
            ground_truth_perplexity = float(math.exp(mean_ground_truth_nll))

    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "generated_text": generated_text,
        "top_tokens": top_tokens,
        "attention_contributions": attention_contributions,
        "ground_truth_token_probabilities": token_probabilities,
        "metrics": {
            "ground_truth_token_count": len(token_probabilities),
            "mean_ground_truth_nll": mean_ground_truth_nll,
            "ground_truth_perplexity": ground_truth_perplexity,
            "top1_token": top_tokens[0]["token"] if top_tokens else "",
            "top1_probability": top_tokens[0]["probability"] if top_tokens else float("nan"),
        },
    }


def finalize_decoder_trace_store(trace_store: Dict[str, List[Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in trace_store.items():
        if not values:
            continue
        if key in {"sample_id"}:
            out[key] = np.asarray(values, dtype=np.int64)
        elif key in {"prompt", "reference", "prediction"}:
            out[key] = np.asarray(values, dtype=object)
        else:
            try:
                out[key] = np.stack(values, axis=0)
            except ValueError:
                out[key] = np.asarray(values, dtype=object)
    return out


def compute_reference_perplexity(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    device: torch.device,
    max_length: int,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reference_ids = tokenizer(reference, add_special_tokens=False)["input_ids"]
    if not reference_ids:
        return float("nan")

    input_ids = prompt_ids + reference_ids
    labels = [-100] * len(prompt_ids) + reference_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    model_inputs = {
        "input_ids": torch.tensor([input_ids], device=get_model_input_device(model)),
        "attention_mask": torch.ones((1, len(input_ids)), device=get_model_input_device(model), dtype=torch.long),
        "labels": torch.tensor([labels], device=get_model_input_device(model)),
    }
    with torch.no_grad():
        loss = model(**model_inputs).loss
    return float(torch.exp(loss).item())


def compute_reference_perplexities_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    references: Sequence[str],
    device: torch.device,
    max_length: int,
) -> List[Dict[str, float]]:
    if not prompts:
        return []

    encoded_inputs: List[List[int]] = []
    encoded_labels: List[List[int]] = []
    model_device = get_model_input_device(model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for prompt, reference in zip(prompts, references):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reference_ids = tokenizer(reference, add_special_tokens=False)["input_ids"]
        if not reference_ids:
            encoded_inputs.append([pad_token_id])
            encoded_labels.append([-100])
            continue

        input_ids = prompt_ids + reference_ids
        labels = [-100] * len(prompt_ids) + reference_ids
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        encoded_inputs.append(input_ids)
        encoded_labels.append(labels)

    batch_width = max(len(input_ids) for input_ids in encoded_inputs)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    for input_ids, labels in zip(encoded_inputs, encoded_labels):
        pad_width = batch_width - len(input_ids)
        padded_input_ids.append(input_ids + [pad_token_id] * pad_width)
        padded_attention_mask.append([1] * len(input_ids) + [0] * pad_width)
        padded_labels.append(labels + [-100] * pad_width)

    if batch_width < 2:
        return [
            {"perplexity": float("nan"), "loss_sum": 0.0, "token_count": 0.0}
            for _ in encoded_inputs
        ]

    model_inputs = {
        "input_ids": torch.tensor(padded_input_ids, device=model_device),
        "attention_mask": torch.tensor(padded_attention_mask, device=model_device, dtype=torch.long),
    }
    labels_tensor = torch.tensor(padded_labels, device=model_device)

    with torch.no_grad():
        logits = model(**model_inputs).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels_tensor[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid_tokens = shift_labels.ne(-100)
    loss_sums = (token_losses * valid_tokens).sum(dim=1)
    token_counts = valid_tokens.sum(dim=1)

    results: List[Dict[str, float]] = []
    for loss_sum, token_count in zip(loss_sums, token_counts):
        if int(token_count.item()) == 0:
            results.append({"perplexity": float("nan"), "loss_sum": 0.0, "token_count": 0.0})
        else:
            mean_loss = loss_sum / token_count
            results.append(
                {
                    "perplexity": float(torch.exp(mean_loss).item()),
                    "loss_sum": float(loss_sum.item()),
                    "token_count": float(token_count.item()),
                }
            )
    return results


def compute_wikitext_corpus_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_length: int,
    stride: int | None = None,
    token_limit: int | None = None,
    device: torch.device | str | None = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, float]:
    resolved_name = _resolve_dataset_name(dataset_name)
    resolved_config = dataset_config or DEFAULT_WIKITEXT_CONFIG
    test = load_dataset(resolved_name, resolved_config, split=split)
    text = "\n\n".join(test["text"])
    if token_limit is not None and token_limit > 0:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=int(token_limit))
    else:
        encodings = tokenizer(text, return_tensors="pt")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    seq_len = encodings.input_ids.size(1)
    eval_stride = int(stride) if stride is not None and stride > 0 else 512
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        for begin_loc in range(0, seq_len, eval_stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    labels=target_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )

            # The model loss is averaged over valid shifted labels, so only the scalar
            # value needs to survive beyond this window.
            num_valid_tokens = target_ids.ne(-100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            nll_sum += float(outputs.loss.detach().cpu().item()) * num_loss_tokens
            n_tokens += num_loss_tokens

            del outputs, input_ids, target_ids

            prev_end_loc = end_loc
            if progress_callback is not None:
                progress_callback(end_loc / max(seq_len, 1), f"Scored {end_loc}/{seq_len} WikiText tokens.")
            if end_loc == seq_len:
                break
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    if n_tokens <= 0:
        return {
            "perplexity": float("nan"),
            "avg_nll": float("nan"),
            "reference_token_count": 0,
            "sequence_token_count": int(seq_len),
        }

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = math.exp(avg_nll)
    return {
        "perplexity": float(ppl),
        "avg_nll": float(avg_nll),
        "reference_token_count": int(n_tokens),
        "sequence_token_count": int(seq_len),
    }


def release_torch_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize_for_text_metrics(text: str) -> List[str]:
    return text.lower().split()


def ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(predictions: Sequence[str], references: Sequence[str], max_order: int = 4) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize_for_text_metrics(prediction)
        ref_tokens = tokenize_for_text_metrics(reference)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(pred_tokens) - order + 1, 0)

    precisions = []
    for idx in range(max_order):
        if possible_matches_by_order[idx] == 0:
            precisions.append(0.0)
        else:
            precisions.append((matches_by_order[idx] + 1.0) / (possible_matches_by_order[idx] + 1.0))

    if min(precisions) <= 0.0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0 if pred_length > ref_length else math.exp(1.0 - (ref_length / max(pred_length, 1)))
    return float(geo_mean * brevity_penalty)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for idx, token_b in enumerate(b, start=1):
            cur = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = cur
    return dp[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_text_metrics(prediction)
    ref_tokens = tokenize_for_text_metrics(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores = [rouge_l_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    return float(sum(scores) / max(len(scores), 1))


def bertscore_pair(
    prediction: str,
    reference: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 512,
) -> Dict[str, float]:
    pred_inputs = tokenizer(prediction, return_tensors="pt", truncation=True, max_length=max_length)
    ref_inputs = tokenizer(reference, return_tensors="pt", truncation=True, max_length=max_length)

    pred_inputs = {key: value.to(device) for key, value in pred_inputs.items()}
    ref_inputs = {key: value.to(device) for key, value in ref_inputs.items()}

    with torch.no_grad():
        pred_hidden = model(**pred_inputs).last_hidden_state[0]
        ref_hidden = model(**ref_inputs).last_hidden_state[0]

    pred_mask = pred_inputs["attention_mask"][0].bool()
    ref_mask = ref_inputs["attention_mask"][0].bool()
    pred_emb = pred_hidden[pred_mask]
    ref_emb = ref_hidden[ref_mask]
    if pred_emb.shape[0] <= 2 or ref_emb.shape[0] <= 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_emb = F.normalize(pred_emb[1:-1], p=2, dim=-1)
    ref_emb = F.normalize(ref_emb[1:-1], p=2, dim=-1)
    similarity = pred_emb @ ref_emb.T
    precision = similarity.max(dim=1).values.mean().item()
    recall = similarity.max(dim=0).values.mean().item()
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
) -> Dict[str, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for prediction, reference in zip(predictions, references):
        score = bertscore_pair(prediction, reference, tokenizer, model, device)
        precisions.append(score["precision"])
        recalls.append(score["recall"])
        f1s.append(score["f1"])

    count = max(len(f1s), 1)
    return {
        "precision": float(sum(precisions) / count),
        "recall": float(sum(recalls) / count),
        "f1": float(sum(f1s) / count),
    }


def infer_dataset_format(dataset_name: str, dataset_format: str = "auto") -> str:
    if dataset_format != "auto":
        return dataset_format
    normalized = dataset_name.lower()
    if "wikitext" in normalized or normalized in WIKITEXT2_ALIASES:
        return "wikitext"
    return "alpaca"


def _resolve_dataset_name(dataset_name: str) -> str:
    if dataset_name.lower() in WIKITEXT2_ALIASES:
        return "wikitext"
    return dataset_name


def _limit_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def _build_alpaca_records(dataset, max_samples: int | None) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for example in _limit_dataset(dataset, max_samples):
        records.append(
            {
                "instruction": str(example.get("instruction", "")),
                "input": str(example.get("input", "")),
                "reference": str(example.get("output", "")),
            }
        )
    return records


def _build_wikitext_records(dataset, max_samples: int | None, text_col: str = "text") -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for example in dataset:
        text = str(example.get(text_col, "")).strip()
        if not text:
            continue

        words = text.split()
        if len(words) >= 8:
            split_at = max(1, min(len(words) - 1, len(words) // 2))
            context = " ".join(words[:split_at])
            reference = " ".join(words[split_at:])
        else:
            context = ""
            reference = text

        records.append(
            {
                "instruction": "Continue the following WikiText passage.",
                "input": context,
                "reference": reference,
            }
        )
        if max_samples is not None and max_samples > 0 and len(records) >= max_samples:
            break
    return records


def build_records(
    dataset_name: str,
    split: str,
    max_samples: int | None,
    dataset_config: str | None = None,
    dataset_format: str = "auto",
    text_col: str = "text",
) -> List[Dict[str, str]]:
    resolved_name = _resolve_dataset_name(dataset_name)
    resolved_config = dataset_config
    resolved_format = infer_dataset_format(resolved_name, dataset_format)
    if resolved_format == "wikitext" and resolved_config is None:
        resolved_config = DEFAULT_WIKITEXT_CONFIG

    dataset = load_dataset(resolved_name, resolved_config, split=split) if resolved_config else load_dataset(resolved_name, split=split)
    if resolved_format == "wikitext":
        return _build_wikitext_records(dataset, max_samples, text_col=text_col)
    return _build_alpaca_records(dataset, max_samples)


def evaluate_decoder_only_model(
    model_name: str,
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str | None = None,
    dataset_format: str = "auto",
    split: str = "train",
    max_samples: int | None = 100,
    batch_size: int = 64,
    perplexity_batch_size: int | None = None,
    perplexity_stride: int | None = None,
    wikitext_token_limit: int | None = None,
    max_input_length: int = 1024,
    max_new_tokens: int = 128,
    device: str | None = None,
    trust_remote_code: bool = False,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_k: int = 40,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
    progress_callback: ProgressCallback = None,
    generation_tokenizer=None,
    generation_model=None,
    trace_enabled: bool = False,
    trace_top_k: int = 10,
) -> Dict[str, Any]:
    resolved_device = resolve_device(device)

    def emit(progress: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(progress, message)

    emit(0.05, "Loading decoder-only model...")
    if generation_tokenizer is None or generation_model is None:
        generation_tokenizer, generation_model = load_generation_model(
            model_name,
            trust_remote_code,
            resolved_device,
        )
    else:
        if not hasattr(generation_model, "hf_device_map"):
            generation_model.to(resolved_device)
        generation_model.eval()

    resolved_dataset_format = infer_dataset_format(_resolve_dataset_name(dataset_name), dataset_format)
    is_wikitext_task = resolved_dataset_format == "wikitext"
    bert_tokenizer = None
    bert_model = None
    if not is_wikitext_task:
        emit(0.20, "Loading BERTScore model...")
        bert_tokenizer, bert_model = load_bertscore_model(bertscore_model, resolved_device)
    emit(0.30, "Loading evaluation dataset...")
    if is_wikitext_task:
        wikitext_stats = compute_wikitext_corpus_perplexity(
            model=generation_model,
            tokenizer=generation_tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_length=max_input_length,
            stride=perplexity_stride,
            token_limit=wikitext_token_limit,
            device=resolved_device,
            progress_callback=lambda frac, msg: emit(0.30 + 0.60 * frac, msg),
        )
        emit(1.0, "Decoder-only evaluation finished.")
        return {
            "config": {
                "model": model_name,
                "dataset": dataset_name,
                "dataset_config": dataset_config or DEFAULT_WIKITEXT_CONFIG,
                "dataset_format": resolved_dataset_format,
                "split": split,
                "max_samples": None,
                "batch_size": batch_size,
                "perplexity_batch_size": None,
                "perplexity_stride": perplexity_stride,
                "wikitext_token_limit": wikitext_token_limit,
                "max_input_length": max_input_length,
                "max_new_tokens": max_new_tokens,
                "device": str(resolved_device),
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "bertscore_model": bertscore_model,
                "trust_remote_code": trust_remote_code,
            },
            "metrics": {
                "perplexity": wikitext_stats["perplexity"],
                "avg_nll": wikitext_stats["avg_nll"],
                "reference_token_count": wikitext_stats["reference_token_count"],
                "sequence_token_count": wikitext_stats["sequence_token_count"],
            },
            "samples": [],
        }

    records = build_records(dataset_name, split, max_samples, dataset_config=dataset_config, dataset_format=dataset_format)

    predictions: List[str] = []
    references: List[str] = []
    perplexities: List[float] = []
    total_reference_loss = 0.0
    total_reference_tokens = 0.0
    sample_outputs: List[Dict[str, Any]] = []
    trace_store: Dict[str, List[Any]] | None = (
        {
            "sample_id": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "generated_token_ids": [],
            "topk_token_ids": [],
            "hidden_by_layer": [],
            "prompt": [],
            "reference": [],
            "prediction": [],
        }
        if trace_enabled
        else None
    )
    total_records = max(len(records), 1)
    eval_batch_size = max(1, int(batch_size))
    eval_perplexity_batch_size = max(
        1,
        int(perplexity_batch_size if perplexity_batch_size is not None else (1 if is_wikitext_task else eval_batch_size)),
    )
    outer_batch_size = eval_perplexity_batch_size if is_wikitext_task else eval_batch_size

    for batch_start in range(0, len(records), outer_batch_size):
        batch_records = records[batch_start:batch_start + outer_batch_size]
        batch_prompts = [
            build_chat_prompt(generation_tokenizer, record["instruction"], record["input"])
            for record in batch_records
        ]
        batch_references = [record["reference"] for record in batch_records]
        if is_wikitext_task:
            batch_predictions = [""] * len(batch_records)
            batch_traces = {}
        elif trace_enabled:
            batch_predictions, batch_traces = generate_responses_batched_with_traces(
                model=generation_model,
                tokenizer=generation_tokenizer,
                prompts=batch_prompts,
                device=resolved_device,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                trace_top_k=trace_top_k,
            )
        else:
            batch_predictions = generate_responses_batched(
                model=generation_model,
                tokenizer=generation_tokenizer,
                prompts=batch_prompts,
                device=resolved_device,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )
            batch_traces = {}
        batch_perplexity_stats: List[Dict[str, float]] = []
        for ppl_start in range(0, len(batch_prompts), eval_perplexity_batch_size):
            ppl_stop = ppl_start + eval_perplexity_batch_size
            batch_perplexity_stats.extend(
                compute_reference_perplexities_batched(
                    model=generation_model,
                    tokenizer=generation_tokenizer,
                    prompts=batch_prompts[ppl_start:ppl_stop],
                    references=batch_references[ppl_start:ppl_stop],
                    device=resolved_device,
                    max_length=max_input_length + max_new_tokens,
                )
            )
            release_torch_cache(resolved_device)

        for offset, (record, prediction, perplexity_stats) in enumerate(zip(batch_records, batch_predictions, batch_perplexity_stats)):
            sample_index = batch_start + offset
            perplexity = perplexity_stats["perplexity"]
            perplexities.append(perplexity)
            total_reference_loss += perplexity_stats["loss_sum"]
            total_reference_tokens += perplexity_stats["token_count"]
            if not is_wikitext_task:
                predictions.append(prediction)
                references.append(record["reference"])
                sample_outputs.append(
                    {
                        "index": sample_index,
                        "instruction": record["instruction"],
                        "input": record["input"],
                        "reference": record["reference"],
                        "prediction": prediction,
                        "reference_perplexity": perplexity,
                    }
                )
            if trace_store is not None:
                trace_store["sample_id"].append(sample_index)
                trace_store["prompt"].append(batch_prompts[offset])
                trace_store["reference"].append(record["reference"])
                trace_store["prediction"].append(prediction)
                for key in ["prompt_input_ids", "prompt_attention_mask", "generated_token_ids", "topk_token_ids", "hidden_by_layer"]:
                    if key in batch_traces:
                        trace_store[key].append(batch_traces[key][offset])

        completed = min(batch_start + len(batch_records), len(records))
        action = "Scored" if is_wikitext_task else "Generated"
        emit(0.30 + 0.55 * (completed / total_records), f"{action} {completed}/{len(records)} samples.")
        release_torch_cache(resolved_device)

    emit(0.90, "Computing perplexity..." if is_wikitext_task else "Computing BLEU, ROUGE-L, perplexity, and BERTScore...")
    corpus_perplexity = (
        float(math.exp(total_reference_loss / total_reference_tokens))
        if total_reference_tokens > 0
        else float("nan")
    )
    if is_wikitext_task:
        metrics = {
            "perplexity": corpus_perplexity,
            "reference_token_count": int(total_reference_tokens),
            "samples_evaluated": len(records),
        }
    else:
        bleu = compute_bleu(predictions, references)
        rouge_l = compute_rouge_l(predictions, references)
        bertscore = compute_bertscore(predictions, references, bert_tokenizer, bert_model, resolved_device)
        metrics = {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "perplexity": corpus_perplexity,
            "mean_sample_perplexity": float(sum(perplexities) / max(len(perplexities), 1)),
            "reference_token_count": int(total_reference_tokens),
            "bertscore_precision": bertscore["precision"],
            "bertscore_recall": bertscore["recall"],
            "bertscore_f1": bertscore["f1"],
        }

    emit(1.0, "Decoder-only evaluation finished.")
    result = {
        "config": {
            "model": model_name,
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "dataset_format": resolved_dataset_format,
            "split": split,
            "max_samples": len(records),
            "batch_size": eval_batch_size,
            "perplexity_batch_size": eval_perplexity_batch_size,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "device": str(resolved_device),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "bertscore_model": bertscore_model,
            "trust_remote_code": trust_remote_code,
        },
        "metrics": metrics,
        "samples": sample_outputs,
    }
    if trace_store is not None:
        result["traces"] = finalize_decoder_trace_store(trace_store)
    return result


def evaluate_model(args: argparse.Namespace) -> Dict[str, Any]:
    max_samples = None if args.all_samples or args.max_samples <= 0 else args.max_samples
    return evaluate_decoder_only_model(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_format=args.dataset_format,
        split=args.split,
        max_samples=max_samples,
        batch_size=args.batch_size,
        perplexity_batch_size=args.perplexity_batch_size,
        perplexity_stride=args.perplexity_stride,
        wikitext_token_limit=None if args.wikitext_token_limit is None or args.wikitext_token_limit <= 0 else args.wikitext_token_limit,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        bertscore_model=args.bertscore_model,
    )


def print_summary(results: Dict[str, Any]) -> None:
    print("Evaluation summary")
    print(json.dumps(results["metrics"], indent=2))


def maybe_write_json(results: Dict[str, Any], output_json: str | None) -> None:
    if output_json is None:
        return
    Path(output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = evaluate_model(args)
    print_summary(results)
    maybe_write_json(results, args.output_json)


if __name__ == "__main__":
    main()
