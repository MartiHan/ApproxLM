import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from approxlm.interfaces.streamlit.config import (
    APPROX_OPTIONS,
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
    refresh_architecture_state,
    run_experiment_backend,
    run_qualitative_backend,
)
from approxlm.application.luts import options_with_current
from approxlm.application.dispatcher import build_dispatcher_summary, load_dispatcher_config
from approxlm.application.analysis import compute_drift_statistics, experiment_display_label, metrics_to_table
from approxlm.adapters.persistence.sqlite import (
    delete_experiment,
    init_db,
    load_experiment_record,
    load_experiments,
    load_recent_experiment_records,
    load_qualitative_evaluations,
    save_experiment,
    save_qualitative_evaluation,
    save_traces_to_npz,
)


DEFAULT_DISPATCHER_CONFIG = "recipes/xlmr_blockwise.yaml"
DISPATCHER_PLOT_ORDER = [f"enc{i}" for i in range(12)] + ["all encs"]
DECODER_DISPATCHER_PLOT_ORDER = [f"dec{i}" for i in range(28)] + ["all decs"]
DEFAULT_WIKI2_PROMPT = "Valkyria Chronicles III is a tactical role-playing video game developed by Sega and Media.Vision for the PlayStation Portable."
DEFAULT_WIKI2_GROUND_TRUTH = " Released in January 2011 in Japan, it is the third game in the Valkyria series."


def _render_matmul_profile(profile: Dict[str, Any]) -> None:
    if not profile:
        return

    summary_df = pd.DataFrame(
        [
            {
                "total_selected_layers": profile.get("total_selected_layers"),
                "total_linear_calls": profile.get("total_linear_calls"),
                "approximate_linear_calls": profile.get("approximate_linear_calls"),
                "exact_linear_calls": profile.get("exact_linear_calls"),
                "total_mac_operations": profile.get("total_mac_operations"),
                "approximate_mac_operations": profile.get("approximate_mac_operations"),
                "exact_mac_operations": profile.get("exact_mac_operations"),
                "approximate_mac_fraction": profile.get("approximate_mac_fraction"),
                "total_scalar_multiplications": profile.get("total_scalar_multiplications"),
                "approximate_scalar_multiplications": profile.get("approximate_scalar_multiplications"),
                "exact_scalar_multiplications": profile.get("exact_scalar_multiplications"),
                "approximate_scalar_fraction": profile.get("approximate_scalar_fraction"),
            }
        ]
    )
    st.markdown("**Forward-pass matmul profile**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    per_layer = profile.get("per_layer", [])
    if per_layer:
        with st.expander("Per-layer matmul workload", expanded=False):
            st.dataframe(pd.DataFrame(per_layer), use_container_width=True, hide_index=True)


def render_result_panel(
    title: str,
    metrics: Dict[str, Any],
    config: Dict[str, Any] | None = None,
    matmul_profile: Dict[str, Any] | None = None,
) -> None:
    st.markdown("**Experiment name:** " + title)
    if config is not None:
        with st.expander("Used configuration", expanded=False):
            st.json(config)
    st.dataframe(metrics_to_table(metrics), use_container_width=True, hide_index=True)
    if matmul_profile:
        _render_matmul_profile(matmul_profile)


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
    architecture: list[dict[str, Any]],
    state_key: str,
    label: str,
) -> None:
    st.session_state.setdefault(state_key, {})
    apply_all = st.checkbox(
        f"Apply {label.lower()} 0 configuration to all {label.lower()}s",
        value=False,
        key=f"apply_all_{state_key}",
        help="Copies matching layer settings from block 0 to later blocks when suffix names match.",
    )

    grouped: Dict[str, list[dict[str, Any]]] = {}
    for item in architecture:
        grouped.setdefault(item["group"], []).append(item)

    ordered_groups = sorted(
        grouped,
        key=lambda group: (
            grouped[group][0].get("block_family") is None,
            grouped[group][0].get("block_family") or "",
            float("inf") if grouped[group][0].get("block_index") is None else grouped[group][0]["block_index"],
            group,
        ),
    )

    with st.container(height=320):
        for group in ordered_groups:
            first_item = grouped[group][0]
            block_index = first_item.get("block_index")
            family = first_item.get("block_family")
            group_label = f"{label} {block_index}" if block_index is not None else group
            with st.expander(group_label, expanded=block_index in (None, 0)):
                for item in grouped[group]:
                    layer_name = item["layer"]
                    current = st.session_state[state_key].get(layer_name, "fp32")
                    disable_item = apply_all and family is not None and block_index not in (None, 0)
                    if disable_item:
                        source_layer = None
                        for source_item in architecture:
                            if (
                                source_item.get("block_family") == family
                                and source_item.get("block_index") == 0
                                and source_item.get("suffix") == item.get("suffix")
                            ):
                                source_layer = source_item["layer"]
                                break
                        if source_layer is not None:
                            st.session_state[state_key][layer_name] = st.session_state[state_key].get(source_layer, "fp32")
                    else:
                        options = options_with_current(APPROX_OPTIONS, current)
                        st.session_state[state_key][layer_name] = st.selectbox(
                            item.get("suffix") or layer_name,
                            options,
                            index=options.index(current) if current in options else 0,
                            key=f"{state_key}_{layer_name}",
                            disabled=disable_item,
                            accept_new_options=True,
                            help="Select a preset or type a LUT name/path. Names are resolved as <name>.npy in the current working directory, then packaged resources.",
                        )


def _refresh_architecture(model_name: str, task_type: str, state_key: str, arch_state_key: str, error_state_key: str) -> None:
    try:
        refresh_architecture_state(
            model_name=model_name,
            task_type=task_type,
            state_key=state_key,
            arch_state_key=arch_state_key,
            error_state_key=error_state_key,
        )
    except Exception as exc:
        st.session_state[error_state_key] = str(exc)


def _architecture_for_state(state_key: str, fallback_architecture: list[dict[str, Any]]) -> list[dict[str, Any]]:
    architecture = st.session_state.get(state_key)
    return architecture if architecture else fallback_architecture


def _execute_and_store_experiment(
    *,
    experiment_name: str,
    config: Dict[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = run_experiment_backend(
        config,
        progress_callback=progress_callback,
        trace_enabled=config.get("trace_enabled", DEFAULT_TRACE_ENABLED),
        attention_mode=config.get("attention_mode", DEFAULT_ATTENTION_MODE),
    )
    traces = result.get("traces", {})
    trace_file_path = save_traces_to_npz(experiment_id, traces) if config.get("trace_enabled") and traces else None
    num_traced_samples = int(len(traces["sample_id"])) if traces and "sample_id" in traces else None

    record = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_url": config["model_url"],
        "dataset_url": config["dataset_url"],
        "config": config,
        "metrics": result.get("metrics", {}),
        "matmul_profile": result.get("matmul_profile", {}),
        "trace_file_path": trace_file_path,
        "trace_enabled": config.get("trace_enabled", False) and trace_file_path is not None,
        "attention_mode": config.get("attention_mode") if config.get("trace_enabled") else None,
        "num_traced_samples": num_traced_samples,
    }
    save_experiment(record)
    record["traces"] = traces
    return record


def _run_experiment(config: Dict[str, Any]) -> None:
    progress_bar = st.container().progress(0, text="Preparing experiment...")

    def update_progress(progress: float, message: str) -> None:
        progress_bar.progress(max(0, min(int(progress * 100), 100)), text=message)

    record = _execute_and_store_experiment(
        experiment_name=st.session_state.exp_name,
        config=config,
        progress_callback=update_progress,
    )
    st.session_state.selected_experiment_id = record["experiment_id"]


def _execute_and_store_qualitative(
    *,
    evaluation_name: str,
    config: Dict[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    evaluation_id = datetime.now().strftime("qual_%Y%m%d_%H%M%S_%f")
    result = run_qualitative_backend(config, progress_callback=progress_callback)
    record = {
        "evaluation_id": evaluation_id,
        "evaluation_name": evaluation_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_url": config["model_url"],
        "prompt": config["prompt"],
        "ground_truth": config["ground_truth"],
        "config": config,
        "result": result,
    }
    save_qualitative_evaluation(record)
    return record


def _run_qualitative(config: Dict[str, Any]) -> None:
    progress_bar = st.container().progress(0, text="Preparing qualitative evaluation...")

    def update_progress(progress: float, message: str) -> None:
        progress_bar.progress(max(0, min(int(progress * 100), 100)), text=message)

    record = _execute_and_store_qualitative(
        evaluation_name=st.session_state.exp_name,
        config=config,
        progress_callback=update_progress,
    )
    st.session_state.latest_qualitative_record = record


def _render_dispatcher_summary(summary: Dict[str, Any]) -> None:
    st.subheader("Dispatcher summary")
    baseline_metrics = summary["baseline_metrics"]
    baseline_bits = ", ".join(f"{metric}={value:.4f}" for metric, value in baseline_metrics.items())
    st.caption(f"Baseline: {summary['baseline_experiment_name']} | {baseline_bits}")
    summary_df = pd.DataFrame(summary["rows"])
    display_columns = ["config_label", "multiplier", "plot_target"]
    for metric_name in summary["metric_names"]:
        if metric_name in summary_df.columns:
            display_columns.append(metric_name)
        drop_name = f"{metric_name}_drop"
        if drop_name in summary_df.columns:
            display_columns.append(drop_name)
    display_columns.extend(["experiment_name", "experiment_id"])
    st.dataframe(summary_df[display_columns], use_container_width=True, hide_index=True)

    plot_df = summary_df[(~summary_df["is_baseline"]) & summary_df["multiplier"].notna() & summary_df["plot_target"].notna()].copy()
    if plot_df.empty:
        return

    categories = DISPATCHER_PLOT_ORDER if summary.get("task_type") == "classification" else DECODER_DISPATCHER_PLOT_ORDER
    plot_df["plot_target"] = pd.Categorical(plot_df["plot_target"], categories=categories, ordered=True)
    plot_df = plot_df.sort_values(["plot_order", "multiplier"])

    for metric_name in summary["metric_names"]:
        if metric_name not in plot_df.columns:
            continue
        baseline_value = baseline_metrics[metric_name]
        baseline_rule_df = pd.DataFrame({"baseline_value": [baseline_value]})
        line = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("plot_target:N", sort=categories, title="Configuration"),
                y=alt.Y(f"{metric_name}:Q", title=metric_name),
                color=alt.Color("multiplier:N", title="Multiplier"),
                tooltip=[
                    "multiplier",
                    "plot_target",
                    alt.Tooltip(f"{metric_name}:Q", format=".4f"),
                    alt.Tooltip(f"{metric_name}_drop:Q", format=".4f"),
                    "experiment_name",
                ],
            )
            .properties(height=260, title=metric_name)
        )
        baseline_rule = (
            alt.Chart(baseline_rule_df)
            .mark_rule(strokeDash=[6, 4], color="#444")
            .encode(y="baseline_value:Q")
        )
        st.altair_chart((line + baseline_rule).resolve_scale(y="shared"), use_container_width=True)


def _run_dispatcher(config_path: str) -> None:
    dispatcher = load_dispatcher_config(config_path)
    experiments = dispatcher.get("experiments", [])
    if not experiments:
        raise RuntimeError("Dispatcher config does not contain any experiments.")

    dispatcher_name = dispatcher.get("dispatcher_name", Path(config_path).stem)
    dispatcher_run_id = datetime.now().strftime("dispatch_%Y%m%d_%H%M%S_%f")
    base_config = dict(dispatcher.get("base_config", {}))
    metrics_to_plot = dispatcher.get("metrics_to_plot")
    records = []
    overall_progress = st.container().progress(0, text=f"Loading dispatcher: {dispatcher_name}")
    total = len(experiments)

    for index, experiment in enumerate(experiments, start=1):
        experiment_name = experiment.get("name") or f"{dispatcher_name}_{index:02d}"
        experiment_label = experiment.get("label") or experiment_name
        run_config = {
            **base_config,
            **{k: v for k, v in experiment.items() if k not in {"name", "label", "tags"}},
            "dispatcher_name": dispatcher_name,
            "dispatcher_run_id": dispatcher_run_id,
            "dispatcher_order": index - 1,
            "dispatcher_experiment_name": experiment_name,
            "dispatcher_experiment_label": experiment_label,
            "dispatcher_tags": list(experiment.get("tags", [])),
        }
        if metrics_to_plot is not None:
            run_config["metrics_to_plot"] = metrics_to_plot

        def update_progress(progress: float, message: str, run_index: int = index, run_label: str = experiment_label) -> None:
            normalized = ((run_index - 1) + max(0.0, min(progress, 1.0))) / max(total, 1)
            overall_progress.progress(
                max(0, min(int(normalized * 100), 100)),
                text=f"[{run_index}/{total}] {run_label} | {message}",
            )

        records.append(
            _execute_and_store_experiment(
                experiment_name=experiment_name,
                config=run_config,
                progress_callback=update_progress,
            )
        )

    overall_progress.progress(100, text=f"Dispatcher finished: {dispatcher_name}")
    st.session_state.dispatcher_summary = build_dispatcher_summary(records)
    st.session_state.selected_experiment_id = records[-1]["experiment_id"]


def _render_dispatcher_panel() -> None:
    st.subheader("Experiment dispatcher")
    st.text_input(
        "Dispatcher config path",
        value=DEFAULT_DISPATCHER_CONFIG,
        key="dispatcher_config_path",
        help="JSON or YAML config describing the experiment set to run sequentially.",
    )
    st.caption("Select a JSON or YAML dispatcher config to run a sequential experiment sweep.")

    if st.button("Run dispatcher", type="primary", use_container_width=True, key="run_dispatcher_button"):
        try:
            _run_dispatcher(st.session_state.dispatcher_config_path)
        except Exception as exc:
            st.error(f"Dispatcher failed: {exc}")

    st.markdown("**Build from saved runs**")
    record_count = st.number_input(
        "Last N database records",
        min_value=1,
        max_value=500,
        value=12,
        step=1,
        key="dispatcher_recent_record_count",
        help="Uses the most recent saved experiments without rerunning them.",
    )
    if st.button("Build dispatcher summary from database", use_container_width=True, key="build_dispatcher_summary_from_db"):
        try:
            records = load_recent_experiment_records(int(record_count))
            if not records:
                st.warning("No saved experiments were found in the database.")
            else:
                st.session_state.dispatcher_summary = build_dispatcher_summary(records)
                st.session_state.selected_experiment_id = records[-1]["experiment_id"]
        except Exception as exc:
            st.error(f"Failed to build dispatcher summary from database: {exc}")

    summary = st.session_state.get("dispatcher_summary")
    if summary:
        _render_dispatcher_summary(summary)


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
    render_result_panel(
        row["experiment_name"],
        json.loads(row["metrics_json"]),
        json.loads(row["config_json"]),
        json.loads(row["matmul_profile_json"]) if row.get("matmul_profile_json") else {},
    )

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
    fallback_architecture = build_xlmr_architecture(12)
    model_box, refresh_box, dataset_box = st.columns([1.4, 0.5, 1.1])
    with model_box:
        model_url = st.text_input("Model URL", value=DEFAULT_MODEL, key="cls_model_url")
    with refresh_box:
        st.write("")
        if st.button("Refresh layers", key="refresh_cls_layers", use_container_width=True):
            _refresh_architecture(
                model_name=model_url,
                task_type="classification",
                state_key="classification_layer_modes",
                arch_state_key="classification_architecture",
                error_state_key="classification_architecture_error",
            )
    with dataset_box:
        dataset_url = st.text_input("Dataset URL", value=DEFAULT_DATASET, key="cls_dataset_url")

    if st.session_state.get("classification_architecture_error"):
        st.warning(f"Failed to inspect model architecture: {st.session_state['classification_architecture_error']}")

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

    architecture = _architecture_for_state("classification_architecture", fallback_architecture)
    _render_layer_selectors(
        architecture=architecture,
        state_key="classification_layer_modes",
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
    fallback_architecture = build_decoder_only_architecture(24)
    col1, col2, col3 = st.columns([1.4, 0.5, 1.1])
    with col1:
        model_url = st.text_input("Decoder-only model", value=DEFAULT_DECODER_MODEL, key="dec_model_url")
    with col2:
        st.write("")
        if st.button("Refresh layers", key="refresh_dec_layers", use_container_width=True):
            _refresh_architecture(
                model_name=model_url,
                task_type="decoder_only",
                state_key="decoder_layer_modes",
                arch_state_key="decoder_architecture",
                error_state_key="decoder_architecture_error",
            )
    with col3:
        dataset_url = st.text_input("Decoder-only dataset", value=DEFAULT_DECODER_DATASET, key="dec_dataset_url")

    if st.session_state.get("decoder_architecture_error"):
        st.warning(f"Failed to inspect model architecture: {st.session_state['decoder_architecture_error']}")

    dec_cfg_1, dec_cfg_2, dec_cfg_3, dec_cfg_4 = st.columns(4)
    with dec_cfg_1:
        dataset_format = st.selectbox(
            "Dataset format",
            ["auto", "alpaca", "wikitext"],
            index=0,
            key="dec_dataset_format",
        )
    with dec_cfg_2:
        dataset_config = st.text_input(
            "Dataset config",
            value="wikitext-2-raw-v1",
            key="dec_dataset_config",
            help="Use wikitext-2-raw-v1 for Hugging Face wikitext.",
        )
    with dec_cfg_3:
        split_name = st.text_input("Split", value="test", key="dec_split_name")
    with dec_cfg_4:
        backend_quantize = st.checkbox(
            "Quantize backend",
            value=True,
            key="dec_backend_quantize",
            help="Disable for already quantized models when backend inputs/weights should be used with scale 1 and zero bias.",
        )

    is_wikitext = dataset_format == "wikitext" or "wikitext" in dataset_url.lower() or dataset_url.lower() in {"wiki2", "wikitext2", "mindchain/wikitext2"}
    sample_min = 0 if is_wikitext else 1
    sample_help = "Ignored for WikiText corpus perplexity; the full split text is tokenized as one corpus."
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        max_samples = st.number_input("Max samples", min_value=sample_min, value=0 if is_wikitext else 100, step=1, help=sample_help if is_wikitext else None)
    with col5:
        max_input_length = st.number_input("Max input length", min_value=64, value=1024, step=64)
    with col6:
        max_new_tokens = st.number_input("Max new tokens", min_value=0, value=0 if is_wikitext else 128, step=16)
    with col7:
        batch_size = st.number_input("Batch size", min_value=1, value=1 if is_wikitext else 4, step=1)

    stride_col, token_col, trust_col, sample_col = st.columns(4)
    with stride_col:
        perplexity_stride = st.number_input(
            "Perplexity stride",
            min_value=0,
            value=512 if is_wikitext else 0,
            step=64,
            help="Notebook default for WikiText is 512.",
        )
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

    architecture = _architecture_for_state("decoder_architecture", fallback_architecture)
    _render_layer_selectors(
        architecture=architecture,
        state_key="decoder_layer_modes",
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


def _render_qualitative_result(record: Dict[str, Any]) -> None:
    result = record.get("result", {})
    metrics = result.get("metrics", {})
    top_tokens = result.get("top_tokens", [])
    attention_contributions = result.get("attention_contributions", [])
    gt_probs = result.get("ground_truth_token_probabilities", [])

    st.markdown("**Qualitative result:** " + record.get("evaluation_name", record.get("evaluation_id", "")))
    if metrics:
        st.dataframe(metrics_to_table(metrics), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prompt**")
        st.write(result.get("prompt", ""))
        st.markdown("**Ground truth**")
        st.write(result.get("ground_truth", ""))
    with col2:
        st.markdown("**Generated continuation**")
        st.write(result.get("generated_text", ""))

    if top_tokens:
        top_df = pd.DataFrame(top_tokens)
        top_df["display_token"] = top_df["token"].map(lambda token: repr(token)[1:-1] if token.strip() != token or token == "" else token)
        chart = (
            alt.Chart(top_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("probability:Q", title="Probability", axis=alt.Axis(format="%")),
                y=alt.Y(
                    "display_token:N",
                    sort="-x",
                    title="Next-token candidate",
                    axis=alt.Axis(labelOverlap=False, labelLimit=240),
                ),
                color=alt.Color("probability:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                tooltip=[
                    alt.Tooltip("rank:O", title="Rank"),
                    alt.Tooltip("token:N", title="Token"),
                    alt.Tooltip("token_id:Q", title="Token ID"),
                    alt.Tooltip("probability:Q", title="Probability", format=".4%"),
                ],
            )
            .properties(height=360, title="Top 10 next-token probabilities")
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(top_df[["rank", "token", "token_id", "probability"]], use_container_width=True, hide_index=True)

    if attention_contributions:
        attention_df = pd.DataFrame(attention_contributions).head(20).copy()
        attention_df = attention_df.sort_values("attention", ascending=True)
        attention_df["display_token"] = attention_df.apply(
            lambda row: f"{row['position']}: {row['token']}",
            axis=1,
        )
        attention_chart = (
            alt.Chart(attention_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("attention_fraction:Q", title="Attention share", axis=alt.Axis(format="%")),
                y=alt.Y(
                    "display_token:N",
                    sort=None,
                    title="Previous token",
                    axis=alt.Axis(labelOverlap=False, labelLimit=260),
                ),
                color=alt.Color("attention_fraction:Q", scale=alt.Scale(scheme="goldgreen"), legend=None),
                tooltip=[
                    alt.Tooltip("position:O", title="Position"),
                    alt.Tooltip("token:N", title="Token"),
                    alt.Tooltip("raw_token:N", title="Raw token"),
                    alt.Tooltip("attention_fraction:Q", title="Attention share", format=".4%"),
                ],
            )
            .properties(height=520, title="Previous-token attention for next prediction")
        )
        st.altair_chart(attention_chart, use_container_width=True)
        strongest = max(attention_contributions, key=lambda item: item.get("attention", 0.0))
        st.caption(
            f"Highest last-layer attention before the next-token prediction: "
            f"{strongest.get('token')} at position {strongest.get('position')}."
        )
    else:
        st.info("Attention weights were not returned by this model/backend for the qualitative prompt.")

    if gt_probs:
        prob_df = pd.DataFrame(gt_probs)
        prob_df["display_token"] = prob_df["token"].map(lambda token: repr(token)[1:-1] if token.strip() != token or token == "" else token)
        heat = (
            alt.Chart(prob_df)
            .mark_rect()
            .encode(
                x=alt.X("position:O", title="Ground-truth token position"),
                y=alt.Y("display_token:N", sort=None, title="Ground-truth token"),
                color=alt.Color("probability:Q", scale=alt.Scale(scheme="viridis"), title="P(token)"),
                tooltip=[
                    alt.Tooltip("position:O", title="Position"),
                    alt.Tooltip("token:N", title="Token"),
                    alt.Tooltip("probability:Q", title="Probability", format=".4%"),
                    alt.Tooltip("surprisal:Q", title="Surprisal", format=".3f"),
                ],
            )
            .properties(height=min(520, 36 + 24 * len(prob_df)), title="Ground-truth token probability trace")
        )
        st.altair_chart(heat, use_container_width=True)


def _render_qualitative_history() -> None:
    history = load_qualitative_evaluations()
    if history.empty:
        return

    st.markdown("**Saved qualitative evaluations**")
    display_rows = []
    for _, row in history.iterrows():
        result = json.loads(row["result_json"])
        metrics = result.get("metrics", {})
        config = json.loads(row["config_json"])
        selected_modes = config.get("layer_modes", {})
        quantized_count = sum(1 for mode in selected_modes.values() if mode not in {None, "None", "fp32"})
        display_rows.append(
            {
                "evaluation_id": row["evaluation_id"],
                "evaluation_name": row["evaluation_name"],
                "created_at": row["created_at"],
                "model_url": row["model_url"],
                "quantized_layers": quantized_count,
                "top1_token": metrics.get("top1_token"),
                "top1_probability": metrics.get("top1_probability"),
                "ground_truth_perplexity": metrics.get("ground_truth_perplexity"),
            }
        )
    display_df = pd.DataFrame(display_rows)
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=180,
    )
    selected_rows = event.selection.get("rows", []) if event else []
    if selected_rows and st.button("Load qualitative result", use_container_width=True):
        selected = history.iloc[selected_rows[0]]
        st.session_state.latest_qualitative_record = {
            "evaluation_id": selected["evaluation_id"],
            "evaluation_name": selected["evaluation_name"],
            "created_at": selected["created_at"],
            "model_url": selected["model_url"],
            "prompt": selected["prompt"],
            "ground_truth": selected["ground_truth"],
            "config": json.loads(selected["config_json"]),
            "result": json.loads(selected["result_json"]),
        }
        st.rerun()

    latest = st.session_state.get("latest_qualitative_record")
    if latest:
        same_sample = history[
            (history["prompt"] == latest.get("prompt"))
            & (history["ground_truth"] == latest.get("ground_truth"))
        ]
        if len(same_sample) > 1:
            compare_rows = []
            for _, row in same_sample.iterrows():
                result = json.loads(row["result_json"])
                metrics = result.get("metrics", {})
                compare_rows.append(
                    {
                        "created_at": row["created_at"],
                        "evaluation_name": row["evaluation_name"],
                        "model_url": row["model_url"],
                        "top1_token": metrics.get("top1_token"),
                        "top1_probability": metrics.get("top1_probability"),
                        "ground_truth_perplexity": metrics.get("ground_truth_perplexity"),
                    }
                )
            st.markdown("**Same-sample comparison**")
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)


def _render_qualitative_tab() -> None:
    fallback_architecture = build_decoder_only_architecture(24)
    col1, col2, col3 = st.columns([1.4, 0.5, 1.1])
    with col1:
        model_url = st.text_input("Decoder-only model", value=DEFAULT_DECODER_MODEL, key="qual_model_url")
    with col2:
        st.write("")
        if st.button("Refresh layers", key="refresh_qual_layers", use_container_width=True):
            _refresh_architecture(
                model_name=model_url,
                task_type="decoder_only",
                state_key="qualitative_layer_modes",
                arch_state_key="qualitative_architecture",
                error_state_key="qualitative_architecture_error",
            )
    with col3:
        dataset_url = st.text_input("Calibration dataset", value=DEFAULT_DECODER_DATASET, key="qual_dataset_url")

    if st.session_state.get("qualitative_architecture_error"):
        st.warning(f"Failed to inspect model architecture: {st.session_state['qualitative_architecture_error']}")

    sample_col, truth_col = st.columns(2)
    with sample_col:
        prompt = st.text_area("Prompt", value=DEFAULT_WIKI2_PROMPT, height=150, key="qual_prompt")
    with truth_col:
        ground_truth = st.text_area("Ground truth continuation", value=DEFAULT_WIKI2_GROUND_TRUTH, height=150, key="qual_ground_truth")

    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        dataset_config = st.text_input("Dataset config", value="wikitext-2-raw-v1", key="qual_dataset_config")
    with cfg2:
        split_name = st.text_input("Split", value="test", key="qual_split_name")
    with cfg3:
        max_input_length = st.number_input("Max input length", min_value=64, value=1024, step=64, key="qual_max_input_length")
    with cfg4:
        max_new_tokens = st.number_input("Max new tokens", min_value=1, value=64, step=8, key="qual_max_new_tokens")

    run1, run2, run3, run4 = st.columns(4)
    with run1:
        backend_quantize = st.checkbox("Quantize backend", value=True, key="qual_backend_quantize")
    with run2:
        trust_remote_code = st.checkbox("Trust remote code", value=False, key="qual_trust_remote_code")
    with run3:
        do_sample = st.checkbox("Sample generation", value=False, key="qual_do_sample")
    with run4:
        calibration_batches = st.number_input("Calibration batches", min_value=1, value=16, step=1, key="qual_calibration_batches")

    architecture = _architecture_for_state("qualitative_architecture", fallback_architecture)
    _render_layer_selectors(
        architecture=architecture,
        state_key="qualitative_layer_modes",
        label="Decoder block",
    )

    if st.button("Run qualitative evaluation", type="primary", use_container_width=True):
        _run_qualitative(
            {
                "task_type": "qualitative_decoder",
                "model_url": model_url,
                "dataset_url": dataset_url,
                "dataset_config": dataset_config.strip() or "wikitext-2-raw-v1",
                "dataset_format": "wikitext",
                "split_name": split_name.strip() or "test",
                "prompt": prompt,
                "ground_truth": ground_truth,
                "layer_modes": st.session_state.qualitative_layer_modes,
                "backend_quantize": backend_quantize,
                "max_input_length": int(max_input_length),
                "max_new_tokens": int(max_new_tokens),
                "batch_size": 1,
                "calibration_batches": int(calibration_batches),
                "calibration_percentile": 99.9,
                "trust_remote_code": trust_remote_code,
                "do_sample": bool(do_sample),
                "temperature": 0.7,
                "top_k": 40,
                "top_token_count": 10,
            }
        )

    latest = st.session_state.get("latest_qualitative_record")
    if latest:
        _render_qualitative_result(latest)
    _render_qualitative_history()


if "exp_name" not in st.session_state:
    st.session_state.exp_name = default_experiment_name()


def main() -> None:
    st.set_page_config(page_title="LLM Approximation Experiments", layout="wide")
    init_db()
    init_state()
    st.session_state.setdefault("classification_layer_modes", {})
    st.session_state.setdefault("decoder_layer_modes", {})
    st.session_state.setdefault("qualitative_layer_modes", {})
    st.session_state.setdefault("latest_qualitative_record", None)

    top_left, top_right = st.columns([1.2, 1.0], gap="large")
    with top_left:
        st.subheader("Experiment")
        st.text_input("Experiment name", key="exp_name")
        tab_cls, tab_dec, tab_qual, tab_dispatch = st.tabs(["Classification", "Decoder-only", "Qualitative evaluation", "Dispatcher"])
        with tab_cls:
            _render_classification_tab()
        with tab_dec:
            _render_decoder_tab()
        with tab_qual:
            _render_qualitative_tab()
        with tab_dispatch:
            _render_dispatcher_panel()

    with top_right:
        _render_results_panel()

    _render_experiment_log()


if __name__ == "__main__":
    main()
