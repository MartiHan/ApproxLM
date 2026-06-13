from __future__ import annotations
import argparse, json
from pathlib import Path
from approxlm.application.experiments import run_experiment
from approxlm.interfaces.yaml.loader import load_experiment_config

def write_json_output(output_json: str, text: str) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')

def main() -> None:
    p=argparse.ArgumentParser(prog='approxlm',description='Evaluate exact and approximate quantized multipliers in language models.')
    p.add_argument('config',help='YAML experiment configuration')
    p.add_argument('--output-json')
    args=p.parse_args(); cfg=load_experiment_config(args.config); result=run_experiment(cfg)
    text=json.dumps(result,indent=2,default=str)
    if args.output_json: write_json_output(args.output_json, text)
    print(text)
if __name__=='__main__': main()
