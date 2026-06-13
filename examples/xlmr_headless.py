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
                per_channel=False,
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


if __name__ == "__main__":
    result = run_experiment(build_config())
    print(json.dumps(result, indent=2, default=str))
