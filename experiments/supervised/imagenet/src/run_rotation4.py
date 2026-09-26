from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from .rotation4_design import MODELS
from .rotation4_pipeline import extract_features, train_probe, train_upstream
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"config.yaml");p.add_argument("--stage",choices=("upstream","features","probes","full"),default="full");a=p.parse_args();c=yaml.safe_load(a.config.read_text())
 if a.stage in ("upstream","full"):
  for seed in c["upstream"]["seeds"]:
   for model in MODELS: train_upstream(ROOT,c,model,int(seed))
 if a.stage in ("features","full"):
  for seed in c["upstream"]["seeds"]:
   for model in MODELS:
    for split in ("train","val"): extract_features(ROOT,c,model,int(seed),split)
 if a.stage in ("probes","full"):
  for seed in c["upstream"]["seeds"]:
   for model in MODELS: train_probe(ROOT,c,model,int(seed))
if __name__=="__main__":main()
