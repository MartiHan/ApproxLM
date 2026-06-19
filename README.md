<div align="center">

# ApproxLM

### Approximate Multiplier Evaluation for Language Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/GUI-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple.svg)]()

_A PyTorch Framework for Evaluating **Approx**imate Multipliers in **L**anguage **M**odels_ 

</div>

---

## Overview

Deploying natural language processing locally on edge devices, such as robotic assistants, is technically challenging because these battery-powered systems have much lower memory, computational and power budgets than the data centres where large language models (LLMs) are typically hosted. LLM inference is dominated by matrix multiplications, whose exact multiplier circuits require considerable chip area, power, and processing time. Selectively replacing exact multipliers with approximate ones can reduce hardware cost, while still achieving acceptable LLM output quality, but the resulting impact of the approximation depends on several inter-related system parameters. Testing every candidate placement directly in chip design would be costly and time-consuming. This work therefore presents a PyTorch-based framework for emulating signed INT8 approximate multipliers in LLMs before their hardware implementation. Validation on XLM-RoBERTa and Qwen2-0.5B shows that circuit-level error metrics alone cannot predict multiplier suitability, demonstrating the need for application-specific software emulations before approaching hardware design implementation.

ApproxLM provides a software emulation layer for evaluating approximate multipliers before committing to FPGA or ASIC implementation. It supports selective replacement of linear-layer multiplications with signed INT8 lookup-table-based approximate multipliers and evaluates the resulting effect on downstream model quality.

The framework is designed for experiments such as:

- layer-wise and block-wise approximation sensitivity analysis,
- exact INT8 versus approximate INT8 comparison,
- operand-order sensitivity analysis for asymmetric multipliers,
- accuracy–hardware cost trade-off analysis,
- qualitative logit and hidden-state drift inspection,
- reproducible batch experiment dispatch from YAML recipes.

## GPU

ApproxLM can run on CPU for small debugging runs, but GPU execution is strongly recommended for realistic experiments. 
The experiments reported in the project were run on an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB of VRAM. 

## Installation

Clone the repository:

```bash
git clone https://github.com/MartiHan/ApproxLM.git
cd ApproxLM
````

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

For the Streamlit interface:

```bash
pip install -e ".[gui]"
```

---

## Quick start: headless Python evaluation

The following example evaluates one selected XLM-RoBERTa layer with an approximate INT8 multiplier LUT.

```python
import json

from approxlm import (
    CalibrationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    QuantizationConfig,
    QuantizerConfig,
    RuntimeConfig,
    TraceConfig,
    run_experiment,
)


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="xlmr_intermediate_mul8s_1KVA",
        model=ModelConfig(
            hf_id="qanastek/XLMRoberta-Alexa-Intents-Classification",
            task_type="classification"
        ),
        dataset=DatasetConfig(
            name="AmazonScience/massive",
            split="test",
            revision="refs/convert/parquet",
            data_dir="en-US",
            text_col="utt",
            label_col="intent",
        ),
        quantization=QuantizationConfig(
            activation=QuantizerConfig(
                format="int8",
                symmetric=True,
                per_channel=True,
            ),
            weight=QuantizerConfig(
                format="int8",
                symmetric=True,
                per_channel=True,
            ),
            calibration=CalibrationConfig(
                method="histogram",
                percentile=99.9,
                batches=50,
            ),
        ),
        runtime=RuntimeConfig(
            batch_size=256,
            max_length=128,
            backend_quantize=True,
        ),
        trace=TraceConfig(enabled=False),
        lut_directory="src/approxlm/resources/luts",
        layer_modes={
            "roberta.encoder.layer.0.intermediate.dense": "mul8s_1KVA",
        },
    )

result = run_experiment(build_config())
print(json.dumps(result, indent=2, default=str))
```

---

## Quick start: YAML dispatcher

For larger ablation studies, define the experiment once and expand layer/multiplier combinations from a YAML recipe.

```bash
approxlm examples/xlmr_headless.yaml --output-json runs/xlmr_result.json
```

Example dispatcher use cases:

```bash
approxlm-dispatch recipes/xlmr_blockwise.yaml
approxlm-dispatch recipes/xlmr_layerwise.yaml
approxlm-dispatch recipes/qwen2_layerwise.yaml
approxlm-dispatch recipes/qwen2_blockwise.yaml
```

---

## Streamlit GUI

Launch the local graphical interface:

```bash
streamlit run streamlit_app.py
```

The GUI supports:

* model and dataset selection,
* automatic layer discovery,
* per-layer selection of FP32, exact INT8, or approximate INT8 execution,
* experiment history stored in SQLite,
* dispatcher execution,
* trace comparison,
* qualitative next-token inspection,
* MAC-profile views.

---

## Hardware synthesis
ApproxLM includes an optional hardware-characterization flow for estimating the cost of exact and approximate multiplier designs. The flow uses:
- [Yosys Open SYnthesis Suite](https://github.com/YosysHQ/yosys) for RTL synthesis and standard-cell technology mapping
- [OpenSTA - Parallax Static Timing Analyzer](https://github.com/parallaxsw/OpenSTA) for static timing and power estimation

**Additional requirements**

The following executables must be installed and available on the system `PATH`:
```bash
yosys --v
sta --v
```

A compatible Liberty standard-cell library is also required. The experiments in this project use the typical-corner [Nangate45 Open Cell Library](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib):
`hardware/pdk/nangate45/NangateOpenCellLibrary_typical.lib`


Then, the Verilog designs of approximate multipliers can be synthesized using the code example below:

The `arithmetic_helpers.v` file provides functional definitions of the `HAX1` (half adder) and `FAX1` (full adder) helper modules used by the multiplier descriptions. These helpers are flattened and remapped to the selected standard-cell library during synthesis.

### Python example

```python
from approxlm.hardware import characterize_verilog

report = characterize_verilog(
    verilog="hardware/rtl/approximate/mul8s_1KR3_pdk45.v",
    extra_sources=[
        "hardware/rtl/helpers/arithmetic_helpers.v",
    ],
    top="mul8s_1KR3",
    liberty=(
        "hardware/pdk/nangate45/"
        "NangateOpenCellLibrary_typical.lib"
    ),
    output_dir="hardware/runs"
)

print(f"Area: {report.area_um2} µm²")
print(f"Maximum delay: {report.max_delay_ns} ns")
print(f"Total power: {report.total_power_uw} µW")
print(f"Cell count: {report.cell_count}")
print(f"Reports: {report.output_dir}")
```

For debugging, the full logs are stored under:
```bash
hardware/runs/<top-module>/ 
├── synthesis.ys 
├── synthesis.log 
├── synthesis_stats.txt 
├── <top-module>_mapped.v 
├── <top-module>_mapped.json 
├── timing.tcl 
├── timing.log 
└── metadata.json
```

**Reported metrics**

| Metric |Interpretation |
|--------|--------------|
|Area | Sum of the areas of the mapped standard cells reported by Yosys |
|Delay | Longest constrained input-to-output data path reported by OpenSTA |
|Power | Liberty-based estimate of internal, switching, and leakage power |
|Cell count |	Number of mapped cells |

Note that the reported area, delay, and power values are pre-layout estimates. They do not include placement, routing, extracted wire parasitics, filler cells, power-grid overhead, or congestion effects.

The absolute values depend on the synthesis tool, Liberty library, process corner, input-drive assumptions, output load, and switching-activity model. Results should therefore be compared only between designs characterized with the same configuration.


## Repository structure

```text
ApproxLM/
├── src/approxlm/
│   ├── domain/          # Experiment, quantization, and layer-policy definitions
│   ├── application/     # Experiment orchestration and dispatcher logic
│   ├── ports/           # Abstract interfaces
│   ├── adapters/        # PyTorch, HuggingFace, persistence, and backend adapters
│   ├── interfaces/      # YAML, CLI, and Streamlit entry points
│   ├── quantization/    # Quantizers and calibrators
│   └── resources/       # LUTs and example dispatcher configs
├── README.md
├── setup.cfg
└── pyproject.toml
```

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
