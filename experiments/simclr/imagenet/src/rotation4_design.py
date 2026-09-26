from __future__ import annotations
import hashlib, json
from pathlib import Path
MODELS=("M1","M2","M3","M4")
def load_json(path): return json.loads(Path(path).read_text())
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
 return h.hexdigest()
def flat_classes(manifest): return [c for g in manifest["groups"] for c in g["classes"]]
def model_wnids(manifest,model): return [c["wnid"] for c in flat_classes(manifest) if model in c["supervised_models"]]
def verify_rotation_manifest(manifest):
 classes=flat_classes(manifest)
 if len(manifest["groups"])!=20 or len(classes)!=100: raise ValueError("Expected 20 groups and 100 classes")
 if any(len(model_wnids(manifest,m))!=80 for m in MODELS): raise ValueError("Every rotation must contain 80 classes")
 return True
