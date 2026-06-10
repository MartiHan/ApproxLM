from __future__ import annotations
import argparse, json
from dataclasses import asdict
from approxlm.application.experiments import run_experiment
from approxlm.interfaces.yaml.loader import load_experiment_config

def main() -> None:
    p=argparse.ArgumentParser(prog='approxlm',description='Evaluate exact and approximate quantized multipliers in language models.')
    p.add_argument('config',help='YAML experiment configuration')
    p.add_argument('--output-json')
    args=p.parse_args(); cfg=load_experiment_config(args.config); result=run_experiment(cfg)
    text=json.dumps(result,indent=2,default=str)
    if args.output_json: open(args.output_json,'w',encoding='utf-8').write(text)
    print(text)
if __name__=='__main__': main()
