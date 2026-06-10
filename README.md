# ApproxLM
_A PyTorch Framework for Evaluating Approximate Multipliers in Language Models_ 
___

Deploying natural language processing locally on edge devices, such as robotic assistants, is technically challenging because these battery-powered systems have much lower memory, computational and power budgets than the data centres where large language models (LLMs) are typically hosted. LLM inference is dominated by matrix multiplications, whose exact multiplier circuits require considerable chip area, power, and processing time. Selectively replacing exact multipliers with approximate ones can reduce hardware cost, while still achieving acceptable LLM output quality, but the resulting impact of the approximation depends on several inter-related system parameters. Testing every candidate placement directly in chip design would be costly and time-consuming. This work therefore presents a PyTorch-based framework for emulating signed INT8 approximate multipliers in LLMs before their hardware implementation. Validation on XLM-RoBERTa and Qwen2-0.5B shows that circuit-level error metrics alone cannot predict multiplier suitability, demonstrating the need for application-specific software emulations before approaching hardware design implementation.

## Quick start

### Installation
Install the headless version of the package:
```aiignore
pip install -e .
```

For GUI version, additionally install the Streamlit packages:
```aiignore
pip install -e ".[gui]"
```

### Headless execution
Running an experiment from a YAML:
```aiignore
axlm examples/xlmr_headless.yaml --output-json result.json
```

Or from a Python:
```aiignore
from axlm import run_experiment
from axlm.interfaces.yaml.loader import load_experiment_config

config = load_experiment_config("examples/xlmr_headless.yaml")
result = run_experiment(config)

print(result)
```


### Streamlit
```aiignore
streamlit run streamlit_app.py
```
