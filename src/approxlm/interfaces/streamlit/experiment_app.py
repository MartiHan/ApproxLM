import json
from datetime import datetime
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from approxlm.interfaces.streamlit.config import (
    DEFAULT_ATTENTION_MODE,
    DEFAULT_DATASET,
    DEFAULT_DECODER_DATASET,
    DEFAULT_DECODER_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TRACE_ENABLED,
    build_decoder_only_architecture,
    build_xlmr_architecture,
    default_experiment_name,
    init_state,
    render_lut_mode_selector,
    run_experiment_backend,
)
from approxlm.application.analysis import compute_drift_statistics, experiment_display_label, metrics_to_table
from approxlm.adapters.persistence.sqlite import (
    delete_experiment,
    init_db,
    load_experiment_record,
    load_experiments,
    save_experiment,
    save_traces_to_npz,
)


def render_result_panel(title: str, metrics: Dict[str, Any], config: Dict[str, Any] | None = None) -> None:
    st.markdown("**Experiment name:** " + title)
    if config is not None:
        with st.expander("Used configuration", expanded=False):
            st.json(config)
    st.dataframe(metrics_to_table(metrics), use_container_width=True, hide_index=True)


def _render_drift_statistics(drift_stats: Dict[str, Any]) -> None:
    st.markdown("**Aligned samples:** " + str(drift_stats["num_aligned_samples"]))
    rendered_blocks = 0

    if "logit_drift" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### Logit drift")
        logit_block = drift_stats["logit_drift"].copy()
        kl_values = logit_block.pop("kl_values", None)
        logit_block.pop("l2_values", None)
        st.dataframe(pd.DataFrame([logit_block]), use_container_width=True, hide_index=True)

        if kl_values:
            chart = (
                alt.Chart(pd.DataFrame({"KL divergence": kl_values}))
                .mark_bar()
                .encode(
                    x=alt.X("KL divergence:Q", bin=alt.Bin(maxbins=30), title="KL divergence"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=["count()"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

    if "cls_embedding_drift" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### CLS embedding drift")
        cls_block = drift_stats["cls_embedding_drift"].copy()
        per_layer_rel_l2 = cls_block.pop("per_layer_mean_rel_l2", None)
        per_layer_cos = cls_block.pop("per_layer_mean_cosine", None)
        st.dataframe(pd.DataFrame([cls_block]), use_container_width=True, hide_index=True)

        if per_layer_rel_l2 is not None:
            st.line_chart(
                pd.DataFrame({"layer": list(range(len(per_layer_rel_l2))), "mean_rel_l2": per_layer_rel_l2}).set_index("layer")
            )
        if per_layer_cos is not None:
            st.line_chart(
                pd.DataFrame({"layer": list(range(len(per_layer_cos))), "mean_cosine_drift": per_layer_cos}).set_index("layer")
            )

    if "topk_token_agreement" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### Top-k token agreement")
        topk_block = drift_stats["topk_token_agreement"].copy()
        per_step_top1 = topk_block.pop("per_step_top1_agreement", None)
        st.dataframe(pd.DataFrame([topk_block]), use_container_width=True, hide_index=True)
        if per_step_top1 is not None:
            st.line_chart(
                pd.DataFrame({"step": list(range(len(per_step_top1))), "top1_agreement": per_step_top1}).set_index("step")
            )

    if "hidden_state_drift" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### Hidden-state drift")
        hidden_block = drift_stats["hidden_state_drift"].copy()
        per_layer_rel_l2 = hidden_block.pop("per_layer_mean_rel_l2", None)
        per_layer_cos = hidden_block.pop("per_layer_mean_cosine", None)
        per_step_rel_l2 = hidden_block.pop("per_step_mean_rel_l2", None)
        st.dataframe(pd.DataFrame([hidden_block]), use_container_width=True, hide_index=True)
        if per_layer_rel_l2 is not None:
            st.line_chart(
                pd.DataFrame({"layer": list(range(len(per_layer_rel_l2))), "mean_rel_l2": per_layer_rel_l2}).set_index("layer")
            )
        if per_layer_cos is not None:
            st.line_chart(
                pd.DataFrame({"layer": list(range(len(per_layer_cos))), "mean_cosine_drift": per_layer_cos}).set_index("layer")
            )
        if per_step_rel_l2 is not None:
            st.line_chart(
                pd.DataFrame({"step": list(range(len(per_step_rel_l2))), "mean_rel_l2": per_step_rel_l2}).set_index("step")
            )

    if "sequence_divergence" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### Sequence divergence")
        st.dataframe(pd.DataFrame([drift_stats["sequence_divergence"]]), use_container_width=True, hide_index=True)

    if "cumulative_generation_drift" in drift_stats:
        rendered_blocks += 1
        st.markdown("#### Cumulative generation drift")
        cumulative_block = drift_stats["cumulative_generation_drift"]
        st.dataframe(
            pd.DataFrame([{"max_generation_steps": cumulative_block.get("max_generation_steps")}]),
            use_container_width=True,
            hide_index=True,
        )
        per_step = cumulative_block.get("per_step_divergence_rate")
        cumulative = cumulative_block.get("cumulative_divergence_rate")
        if per_step is not None and cumulative is not None:
            st.line_chart(
                pd.DataFrame(
                    {
                        "step": list(range(len(per_step))),
                        "per_step_divergence": per_step,
                        "cumulative_divergence": cumulative,
                    }
                ).set_index("step")
            )

    if rendered_blocks == 0:
        st.warning("The selected traces aligned by sample_id, but no common drift metric arrays were found.")


def _render_layer_selectors(
    architecture: list[dict[str, str]],
    state_key: str,
    group_prefix: str,
    layer_prefix: str,
    block_count: int,
    label: str,
) -> None:
    st.session_state.setdefault(state_key, {})
    apply_all = st.checkbox(
        f"Apply {label.lower()} 0 configuration to all {label.lower()}s",
        value=False,
        key=f"apply_all_{state_key}",
    )

    with st.container(height=320):
        for block_idx in range(block_count):
            with st.expander(f"{label} {block_idx}", expanded=block_idx == 0):
                for item in architecture:
                    if item["group"] != f"{group_prefix}_{block_idx}":
                        continue
                    layer_name = item["layer"]
                    current = st.session_state[state_key].get(layer_name, "fp32")
                    if apply_all and block_idx > 0:
                        source_name = layer_name.replace(f".{block_idx}.", ".0.", 1)
                        st.session_state[state_key][layer_name] = st.session_state[state_key].get(source_name, "fp32")
                    else:
                        st.session_state[state_key][layer_name] = render_lut_mode_selector(
                            layer_name.split(f"{layer_prefix}.{block_idx}.")[-1],
                            key=f"{state_key}_{layer_name}",
                            current=current,
                            disabled=apply_all and block_idx > 0,
                        )


def _run_experiment(config: Dict[str, Any]) -> None:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    progress_bar = st.container().progress(0, text="Preparing experiment...")

    def update_progress(progress: float, message: str) -> None:
        progress_bar.progress(max(0, min(int(progress * 100), 100)), text=message)

    result = run_experiment_backend(
        config,
        progress_callback=update_progress,
        trace_enabled=config.get("trace_enabled", DEFAULT_TRACE_ENABLED),
        attention_mode=config.get("attention_mode", DEFAULT_ATTENTION_MODE),
    )
    traces = result.get("traces", {})
    trace_file_path = save_traces_to_npz(experiment_id, traces) if config.get("trace_enabled") and traces else None
    num_traced_samples = int(len(traces["sample_id"])) if traces and "sample_id" in traces else None

    save_experiment(
        {
            "experiment_id": experiment_id,
            "experiment_name": st.session_state.exp_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_url": config["model_url"],
            "dataset_url": config["dataset_url"],
            "config": config,
            "metrics": result.get("metrics", {}),
            "trace_file_path": trace_file_path,
            "trace_enabled": config.get("trace_enabled", False) and trace_file_path is not None,
            "attention_mode": config.get("attention_mode") if config.get("trace_enabled") else None,
            "num_traced_samples": num_traced_samples,
        }
    )
    st.session_state.selected_experiment_id = experiment_id


def _render_results_panel() -> None:
    st.subheader("Results")
    experiments_df = load_experiments()
    selected_id = st.session_state.selected_experiment_id
    if not selected_id or experiments_df.empty:
        st.info("Run or select an experiment to see results.")
        return

    selected_row = experiments_df.loc[experiments_df["experiment_id"] == selected_id]
    if selected_row.empty:
        st.info("Run or select an experiment to see results.")
        return

    row = selected_row.iloc[0]
    render_result_panel(row["experiment_name"], json.loads(row["metrics_json"]), json.loads(row["config_json"]))

    st.subheader("Experiment comparison")
    trace_path = experiments_df["trace_file_path"] if "trace_file_path" in experiments_df else pd.Series("", index=experiments_df.index)
    traceable_mask = experiments_df["trace_enabled"].fillna(0).astype(bool) & trace_path.notna() & trace_path.astype(str).str.strip().ne("")
    compare_df = experiments_df.loc[traceable_mask, ["experiment_id", "experiment_name", "created_at"]].copy()
    if len(compare_df) < 2:
        st.info("Drift comparison needs two experiments with saved trace artifacts. Rerun the experiments you want to compare with tracing enabled.")
        return

    compare_df["display"] = compare_df.apply(experiment_display_label, axis=1)

    col1, col2 = st.columns(2)
    with col1:
        baseline_display = st.selectbox("Baseline experiment", options=compare_df["display"].tolist(), key="baseline_compare_select")
    with col2:
        compare_display = st.selectbox("Experiment to compare", options=compare_df["display"].tolist(), key="target_compare_select")

    if st.button("Compute drift statistics", use_container_width=True):
        base_id = compare_df.loc[compare_df["display"] == baseline_display, "experiment_id"].iloc[0]
        cmp_id = compare_df.loc[compare_df["display"] == compare_display, "experiment_id"].iloc[0]
        if base_id == cmp_id:
            st.warning("Please select two different experiments.")
            return

        try:
            drift_stats = compute_drift_statistics(
                load_experiment_record(base_id, load_traces=True),
                load_experiment_record(cmp_id, load_traces=True),
            )
            _render_drift_statistics(drift_stats)
        except Exception as exc:
            st.error(f"Failed to compute drift statistics: {exc}")


def _render_experiment_log() -> None:
    st.subheader("Experiment log")
    experiments_df = load_experiments()
    if experiments_df.empty:
        st.info("No experiments logged yet.")
        return

    log_df = experiments_df[["experiment_id", "experiment_name", "created_at", "model_url", "dataset_url"]]
    event = st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=180,
    )
    selected_rows = event.selection.get("rows", []) if event else []
    if not selected_rows:
        return

    selected_id = log_df.iloc[selected_rows[0]]["experiment_id"]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load selected", use_container_width=True):
            st.session_state.selected_experiment_id = selected_id
            st.rerun()
    with col2:
        if st.button("Delete selected", use_container_width=True, type="secondary"):
            delete_experiment(selected_id)
            if st.session_state.selected_experiment_id == selected_id:
                st.session_state.selected_experiment_id = None
            st.rerun()


def _render_classification_tab() -> None:
    architecture = build_xlmr_architecture(12)
    model_box, dataset_box = st.columns(2)
    with model_box:
        model_url = st.text_input("Model URL", value=DEFAULT_MODEL, key="cls_model_url")
    with dataset_box:
        dataset_url = st.text_input("Dataset URL", value=DEFAULT_DATASET, key="cls_dataset_url")

    backend_quantize = st.checkbox(
        "Quantize before backend replacement",
        value=True,
        key="cls_backend_quantize",
        help="When disabled, selected backend layers use scale 1 and zero bias instead of calibration/requantization.",
    )
    trace_enabled = st.checkbox(
        "Record traces for drift statistics",
        value=DEFAULT_TRACE_ENABLED,
        key="cls_trace_enabled",
        help="Stores per-sample trace artifacts so this run can be used in drift-statistics comparisons.",
    )

    _render_layer_selectors(
        architecture=architecture,
        state_key="classification_layer_modes",
        group_prefix="encoder",
        layer_prefix="roberta.encoder.layer",
        block_count=12,
        label="Encoder block",
    )

    if st.button("Run classification experiment", type="primary", use_container_width=True):
        _run_experiment(
            {
                "task_type": "classification",
                "model_url": model_url,
                "dataset_url": dataset_url,
                "layer_modes": st.session_state.classification_layer_modes,
                "backend_quantize": backend_quantize,
                "trace_enabled": trace_enabled,
                "attention_mode": DEFAULT_ATTENTION_MODE,
            }
        )


def _render_decoder_tab() -> None:
    architecture = build_decoder_only_architecture(24)
    col1, col2 = st.columns(2)
    with col1:
        model_url = st.text_input("Decoder-only model", value=DEFAULT_DECODER_MODEL, key="dec_model_url")
    with col2:
        dataset_url = st.text_input("Decoder-only dataset", value=DEFAULT_DECODER_DATASET, key="dec_dataset_url")

    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        dataset_format = st.selectbox("Dataset format", ["auto", "alpaca", "wikitext"], index=0, key="dec_dataset_format")
    with cfg2:
        dataset_config = st.text_input("Dataset config", value="wikitext-2-raw-v1", key="dec_dataset_config")
    with cfg3:
        split_name = st.text_input("Split", value="test", key="dec_split_name")
    with cfg4:
        backend_quantize = st.checkbox(
            "Quantize backend",
            value=True,
            key="dec_backend_quantize",
            help="Disable for already quantized models when backend inputs/weights should be used with scale 1 and zero bias.",
        )

    is_wikitext = dataset_format == "wikitext" or "wikitext" in dataset_url.lower() or dataset_url.lower() in {"wiki2", "wikitext2", "mindchain/wikitext2"}
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        max_samples = st.number_input("Max samples", min_value=0 if is_wikitext else 1, value=0 if is_wikitext else 100, step=1)
    with col4:
        max_input_length = st.number_input("Max input length", min_value=64, value=1024, step=64)
    with col5:
        max_new_tokens = st.number_input("Max new tokens", min_value=0, value=0 if is_wikitext else 128, step=16)
    with col6:
        batch_size = st.number_input("Batch size", min_value=1, value=1 if is_wikitext else 4, step=1)

    stride_col, token_col, trust_col, sample_col = st.columns(4)
    with stride_col:
        perplexity_stride = st.number_input("Perplexity stride", min_value=0, value=512 if is_wikitext else 0, step=64)
    with token_col:
        wikitext_token_limit = st.number_input(
            "Max WikiText tokens",
            min_value=0,
            value=0,
            step=1024,
            disabled=not is_wikitext,
            help="Number of tokenized split tokens to evaluate for WikiText. Use 0 for the full split.",
        )
    with trust_col:
        trust_remote_code = st.checkbox("Trust remote code", value=False, key="dec_trust_remote_code")
    with sample_col:
        do_sample = st.checkbox("Sample generation", value=False, key="dec_do_sample", disabled=is_wikitext)

    trace_enabled = st.checkbox(
        "Record traces for drift statistics",
        value=DEFAULT_TRACE_ENABLED and not is_wikitext,
        key="dec_trace_enabled",
        disabled=is_wikitext,
        help="Stores decoder trace artifacts for drift-statistics comparisons. WikiText corpus runs do not produce comparable per-sample traces.",
    )

    _render_layer_selectors(
        architecture=architecture,
        state_key="decoder_layer_modes",
        group_prefix="decoder",
        layer_prefix="model.layers",
        block_count=24,
        label="Decoder block",
    )

    if st.button("Run decoder-only experiment", type="primary", use_container_width=True):
        _run_experiment(
            {
                "task_type": "decoder_only",
                "model_url": model_url,
                "dataset_url": dataset_url,
                "dataset_config": dataset_config.strip() or None,
                "dataset_format": dataset_format,
                "split_name": split_name.strip() or ("test" if is_wikitext else "train"),
                "layer_modes": st.session_state.decoder_layer_modes,
                "backend_quantize": backend_quantize,
                "max_samples": int(max_samples),
                "max_input_length": int(max_input_length),
                "max_new_tokens": int(max_new_tokens),
                "batch_size": int(batch_size),
                "perplexity_stride": None if int(perplexity_stride) <= 0 else int(perplexity_stride),
                "wikitext_token_limit": None if int(wikitext_token_limit) <= 0 else int(wikitext_token_limit),
                "calibration_batches": 16,
                "calibration_percentile": 99.9,
                "trust_remote_code": trust_remote_code,
                "do_sample": bool(do_sample and not is_wikitext),
                "temperature": 0.7,
                "top_k": 40,
                "bertscore_model": "bert-base-uncased",
                "trace_enabled": bool(trace_enabled and not is_wikitext),
                "attention_mode": None,
            }
        )


if "exp_name" not in st.session_state:
    st.session_state.exp_name = default_experiment_name()


def main() -> None:
    st.set_page_config(page_title="LLM Approximation Experiments", layout="wide")
    init_db()
    init_state()
    st.session_state.setdefault("classification_layer_modes", {})
    st.session_state.setdefault("decoder_layer_modes", {})

    top_left, top_right = st.columns([1.2, 1.0], gap="large")
    with top_left:
        st.subheader("Experiment")
        st.text_input("Experiment name", key="exp_name")
        tab_cls, tab_dec = st.tabs(["Classification", "Decoder-only"])
        with tab_cls:
            _render_classification_tab()
        with tab_dec:
            _render_decoder_tab()

    with top_right:
        _render_results_panel()

    _render_experiment_log()


if __name__ == "__main__":
    main()
