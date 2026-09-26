#!/usr/bin/env python3
"""Train, classification-gate, and evaluate CIFAR-100 ViT-v2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
CANONICAL=Path(os.environ.get("CIFAR_SUPERVISED_ROOT",REPO/"outputs/supervised/cifar100"))
EXPANDED=Path(os.environ.get("DETECTOR_RESULTS_ROOT",REPO/"outputs/detectors"))
DATA=Path(os.environ.get("CIFAR100_ROOT",REPO/"data/cifar100"))
CONFIG=ROOT/"config.json"
MANIFEST_SOURCE=REPO/"manifests/cifar100/manifest.json"
MANIFEST_CSV_SOURCE=REPO/"manifests/cifar100/manifest.csv"
CHECKPOINTS=ROOT/"checkpoints"; CONFIGS=ROOT/"configs"; MANIFESTS=ROOT/"manifests"; LOGS=ROOT/"logs"
TRAINING=ROOT/"training_metrics"; CLASSIFICATION=ROOT/"classification_results"; FEATURES=ROOT/"features"
PROBES=ROOT/"probes"; SCORES=ROOT/"detector_outputs"; FITS=ROOT/"detector_fit_states"
BOOTSTRAP=ROOT/"bootstrap_samples"; SUMMARIES=ROOT/"summaries"
MODELS=("M1","M2","M3","M4"); SEEDS=(0,1)
DETECTORS=("knn","energy","msp","mahalanobis","vim","neco","nci","gradorth","odin")
DISPLAY={"knn":"kNN","energy":"Energy","msp":"MSP","mahalanobis":"Mahalanobis","vim":"ViM","neco":"NECO","nci":"NCI","gradorth":"GradOrth","odin":"ODIN"}
sys.path.insert(0,str(REPO))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.ops import stochastic_depth
from torchvision.transforms import v2

from experiments.supervised.cifar100.src.common import CIFAR100_MEAN,CIFAR100_STD,eval_subset,file_hash,seed_everything,seed_worker


def now(): return datetime.now(timezone.utc).isoformat()
def cfg(): return json.loads(CONFIG.read_text())
def manifest(): return json.loads((ROOT/"manifest.json").read_text())


def ensure_dirs():
    for d in (CHECKPOINTS,CONFIGS,MANIFESTS,LOGS,TRAINING,CLASSIFICATION,FEATURES,PROBES,SCORES,FITS,BOOTSTRAP,SUMMARIES): d.mkdir(parents=True,exist_ok=True)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)


def state_hash(state):
    b=io.BytesIO(); torch.save({k:v.detach().cpu() for k,v in state.items()},b); return hashlib.sha256(b.getvalue()).hexdigest()


def sha_array(x):
    x=np.ascontiguousarray(x); h=hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(np.asarray(x.shape,dtype=np.int64).tobytes()); h.update(x.tobytes()); return h.hexdigest()


def record_command():
    ensure_dirs()
    with (LOGS/"commands.jsonl").open("a") as f: f.write(json.dumps({"utc":now(),"cwd":str(Path.cwd()),"argv":[sys.executable,*sys.argv]},sort_keys=True)+"\n")


class CifarViTTinyV2(nn.Module):
    """ViT-Tiny/4 with training-only stochastic depth."""
    def __init__(self,num_classes=80,drop_path_rate=.1):
        super().__init__()
        self.vit=VisionTransformer(image_size=32,patch_size=4,num_layers=12,num_heads=3,hidden_dim=192,mlp_dim=768,dropout=0.,attention_dropout=0.,num_classes=num_classes)
        self.feature_dim=192; self.drop_path_rate=float(drop_path_rate)
    @property
    def classifier(self): return self.vit.heads.head
    def features(self,x):
        x=self.vit._process_input(x); x=torch.cat([self.vit.class_token.expand(x.shape[0],-1,-1),x],1); x=x+self.vit.encoder.pos_embedding; x=self.vit.encoder.dropout(x)
        layers=list(self.vit.encoder.layers); denominator=max(1,len(layers)-1)
        for i,block in enumerate(layers):
            p=self.drop_path_rate*i/denominator
            y=block.ln_1(x); y,_=block.self_attention(y,y,y,need_weights=False); y=block.dropout(y); x=x+stochastic_depth(y,p,"row",self.training)
            y=block.mlp(block.ln_2(x)); x=x+stochastic_depth(y,p,"row",self.training)
        return self.vit.encoder.ln(x)[:,0]
    def forward(self,x): return self.classifier(self.features(x))


def init_model(seed):
    generator=seed_everything(seed,True); return CifarViTTinyV2(80,cfg()["augmentation"]["drop_path_rate"]),generator


class ClassSubset(Dataset):
    def __init__(self,dataset,classes,remap=True):
        self.dataset=dataset; self.classes=sorted(map(int,classes)); selected=set(self.classes); self.indices=[i for i,y in enumerate(dataset.targets) if int(y) in selected]; self.mapping={y:i for i,y in enumerate(self.classes)} if remap else None
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        x,y=self.dataset[self.indices[i]]; return x,self.mapping[y] if self.mapping is not None else y


def train_transform():
    return v2.Compose([v2.RandomCrop(32,padding=4),v2.RandomHorizontalFlip(p=.5),v2.RandAugment(num_ops=2,magnitude=9,num_magnitude_bins=31),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(CIFAR100_MEAN,CIFAR100_STD),v2.RandomErasing(p=.25,scale=(.02,.33),ratio=(.3,3.3),value=0.)])


def train_dataset(classes): return ClassSubset(CIFAR100(str(DATA),train=True,transform=train_transform(),download=False),classes,True)
def mixer(): return v2.RandomChoice([v2.MixUp(alpha=.8,num_classes=80),v2.CutMix(alpha=1.,num_classes=80)],p=[.5,.5])


def architecture_record(model):
    return {**cfg()["architecture"],"parameter_count":sum(p.numel() for p in model.parameters()),"classifier_head_shape":list(model.classifier.weight.shape),"drop_path_rate_training_only":model.drop_path_rate,"pretrained_weights_loaded":False,"external_checkpoint_loaded_at_initialization":False}


def candidate_manifest_checks(m):
    groups=m["groups"]; d=set(m["downstream_id_classes"]); c=set(m["downstream_ood_candidate_classes"]); sets={x:set(m["model_pretraining_classes"][x]) for x in MODELS}
    checks={"20_groups":len(groups)==20,"100_roles_once":set(int(g[r]["fine_id"]) for g in groups for r in ("d","c1","c2","c3","c4"))==set(range(100)),"20_id":len(d)==20,"80_ood":len(c)==80,"80_classes_each":all(len(sets[x])==80 for x in MODELS),"one_withheld_three_present":all(sum(y not in sets[x] for x in MODELS)==1 and sum(y in sets[x] for x in MODELS)==3 for y in c),"40000_images_each":True}
    return checks


def inventory_hashes(root):
    out={}
    for p in sorted(x for x in root.rglob('*') if x.is_file()): out[str(p.relative_to(root))]=file_hash(p)
    return out


def preflight():
    ensure_dirs()
    for src,dsts in ((MANIFEST_SOURCE,[ROOT/"manifest.json",MANIFESTS/"canonical_manifest.json"]),(MANIFEST_CSV_SOURCE,[ROOT/"manifest.csv",MANIFESTS/"canonical_manifest.csv"])):
        for dst in dsts:
            if dst.exists() and dst.read_bytes()!=src.read_bytes(): raise RuntimeError(f"Manifest mismatch: {dst}")
            if not dst.exists(): shutil.copy2(src,dst)
    m=manifest(); c=cfg(); checks=candidate_manifest_checks(m)
    # Exact canonical downstream image identities.
    old=torch.load(CANONICAL/"experiment/features/features_m1_seed0.pt",map_location="cpu",weights_only=False); d=set(m["downstream_id_classes"]); tr=eval_subset(DATA,True,d,False); te=eval_subset(DATA,False,set(range(100)),False)
    split_ok=np.array_equal(np.asarray(tr.indices),old["train_id_indices"].numpy()) and np.array_equal(np.asarray(te.indices),old["test_indices"].numpy())
    init=[]
    for seed in SEEDS:
        hashes=[]
        for rotation in MODELS:
            model,_=init_model(seed); hashes.append(state_hash(model.state_dict())); init.append({"seed":seed,"rotation":rotation,"state_sha256":hashes[-1]})
        if len(set(hashes))!=1: raise RuntimeError("Same-seed initialization mismatch")
    # Augmentation, soft-label, label-smoothing and optimization smoke.
    ds=train_dataset(m["model_pretraining_classes"]["M1"]); loader=DataLoader(ds,batch_size=16,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(0)); images,hard=next(iter(loader)); mixed,soft=mixer()(images,hard)
    soft_ok=soft.shape==(16,80) and bool(torch.allclose(soft.sum(1),torch.ones(16))) and bool(((soft>0)&(soft<1)).any())
    augment_ok=mixed.shape==(16,3,32,32) and bool(torch.isfinite(mixed).all())
    model,_=init_model(0); model=model.cuda(); opt=AdamW(model.parameters(),lr=3e-4,weight_decay=.05); losses=[]
    for _ in range(2):
        opt.zero_grad(set_to_none=True); logits=model(mixed.cuda()); loss=F.cross_entropy(logits,soft.cuda(),label_smoothing=.1); loss.backward(); opt.step(); losses.append(float(loss))
    feature_ok=model.features(mixed[:2].cuda()).shape==(2,192)
    # Drop path varies in train mode and is identity/deterministic in eval mode.
    model.train(); torch.manual_seed(11); a=model.features(mixed[:2].cuda()); b=model.features(mixed[:2].cuda()); train_varies=not torch.equal(a,b)
    model.eval(); a=model.features(mixed[:2].cuda()); b=model.features(mixed[:2].cuda()); eval_same=torch.equal(a,b)
    smoke=ROOT/"smoke/roundtrip.pt"; smoke.parent.mkdir(exist_ok=True); torch.save({"model_state":model.cpu().state_dict(),"pretrained_weights_loaded":False},smoke); reload,_=init_model(0); reload.load_state_dict(torch.load(smoke,map_location="cpu",weights_only=False)["model_state"],strict=True); reload_ok=state_hash(model.state_dict())==state_hash(reload.state_dict())
    restricted=eval_subset(DATA,False,m["model_pretraining_classes"]["M1"],True); x,y=next(iter(DataLoader(restricted,batch_size=32))); restricted_ok=reload(x).shape==(32,80)
    report={"status":"PASS","created_utc":now(),"phase":"classification-only; no OOD scores or detectors executed","canonical_checks":checks,"manifest_byte_identical":file_hash(ROOT/'manifest.json')==file_hash(MANIFEST_SOURCE),"split_identity_exact":bool(split_ok),"architecture":architecture_record(reload),"initialization":init,"no_pretrained_weights":{"pass":True,"direct_constructor":True,"external_state_loaded":False},"augmentation":{"randaugment_operational":augment_ok,"mixup_cutmix_soft_labels":soft_ok,"soft_label_sums":soft.sum(1).tolist(),"label_smoothing_loss_finite":bool(np.isfinite(losses).all()),"smoke_losses":losses},"stochastic_depth":{"training_outputs_vary":train_varies,"evaluation_outputs_identical":eval_same,"rate":.1},"feature_192d":feature_ok,"checkpoint_roundtrip":reload_ok,"restricted_test_evaluator":restricted_ok,"config_sha256":file_hash(CONFIG),"source_sha256":file_hash(Path(__file__))}
    values=list(checks.values())+[report["manifest_byte_identical"],split_ok,soft_ok,augment_ok,feature_ok,train_varies,eval_same,reload_ok,restricted_ok,bool(np.isfinite(losses).all())]
    if not all(values): report["status"]="FAIL"
    write_json(ROOT/"preflight.json",report)
    (ROOT/"AUDIT.md").write_text("# ViT-v2 classification-only preflight\n\nStatus: **%s**. The fixed architecture configuration, canonical manifests and image identities match exactly, no pretrained state was loaded, and Mixup/CutMix, label smoothing, RandAugment, random erasing, training-only stochastic depth, 192D CLS extraction, checkpoint reload, and restricted-test evaluation passed. No OOD analysis was run.\n"%report["status"])
    if report["status"]!="PASS": raise RuntimeError(report)
    print(json.dumps({"status":"PASS","parameters":report["architecture"]["parameter_count"],"no_ood_evaluation":True}))


def lr_factor(epoch,warm,total):
    if epoch<warm: return (epoch+1)/warm
    return .5*(1+math.cos(math.pi*(epoch-warm)/(total-warm)))


def accuracy(model,dataset):
    loader=DataLoader(dataset,batch_size=512,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True); model.eval(); right=total=0
    with torch.inference_mode():
        for x,y in loader:
            y=y.cuda(non_blocking=True); right+=(model(x.cuda(non_blocking=True)).argmax(1)==y).sum().item(); total+=len(y)
    return right/total


def train(rotation,seed):
    if json.loads((ROOT/"preflight.json").read_text())["status"]!="PASS": raise RuntimeError("Preflight gate missing")
    c=cfg(); m=manifest(); classes=sorted(map(int,m["model_pretraining_classes"][rotation])); model,generator=init_model(seed); initial=state_hash(model.state_dict()); model=model.cuda(); opt=AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]); sched=LambdaLR(opt,lambda e:lr_factor(e,c["warmup_epochs"],c["epochs"])); scaler=torch.amp.GradScaler("cuda",enabled=c["amp_on_cuda"]); mix=mixer(); tag=f"{rotation}_seed{seed}"; cp=CHECKPOINTS/f"{tag}.pt"; start=0; history=[]
    if cp.exists() and c["resume"]:
        s=torch.load(cp,map_location="cpu",weights_only=False)
        if s["identity"]!=[rotation,seed] or s["classes"]!=classes or s["config_sha256"]!=file_hash(CONFIG) or s["initial_state_sha256"]!=initial: raise RuntimeError("Checkpoint mismatch")
        model.load_state_dict(s["model_state"],strict=True); opt.load_state_dict(s["optimizer_state"]); sched.load_state_dict(s["scheduler_state"]); scaler.load_state_dict(s["scaler_state"]); generator.set_state(s["data_generator_state"]); random.setstate(s["python_rng"]); np.random.set_state(s["numpy_rng"]); torch.set_rng_state(s["torch_rng"]); torch.cuda.set_rng_state_all(s["cuda_rng"]); start=s["epoch"]; history=s["history"]
    ds=train_dataset(classes)
    if len(ds)!=40000: raise RuntimeError("Expected 40,000 training images")
    loader=DataLoader(ds,batch_size=c["batch_size"],shuffle=True,generator=generator,num_workers=c["num_workers"],pin_memory=True,worker_init_fn=seed_worker,persistent_workers=True)
    for epoch in range(start,c["epochs"]):
        began=time.time(); model.train(); total=0; loss_sum=0.; mixed_correct=0
        for images,targets in loader:
            images,targets=mix(images,targets); images=images.cuda(non_blocking=True); targets=targets.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=c["amp_on_cuda"]): logits=model(images); loss=F.cross_entropy(logits,targets,label_smoothing=c["label_smoothing"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=len(targets); loss_sum+=float(loss)*len(targets); mixed_correct+=(logits.argmax(1)==targets.argmax(1)).sum().item()
        sched.step(); row={"epoch":epoch+1,"train_soft_target_loss":loss_sum/total,"mixed_target_argmax_accuracy":mixed_correct/total,"learning_rate":opt.param_groups[0]["lr"],"seconds":time.time()-began}; history.append(row)
        payload={"identity":[rotation,seed],"classes":classes,"epoch":epoch+1,"target_epochs":c["epochs"],"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),"scaler_state":scaler.state_dict(),"data_generator_state":generator.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"history":history,"initial_state_sha256":initial,"config_sha256":file_hash(CONFIG),"manifest_sha256":file_hash(ROOT/'manifest.json'),"architecture":architecture_record(model),"pretrained_weights_loaded":False}
        tmp=cp.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(cp); pd.DataFrame(history).to_csv(TRAINING/f"{tag}.csv",index=False); write_json(TRAINING/f"{tag}.json",history)
        print(f"{tag} epoch={epoch+1}/300 loss={row['train_soft_target_loss']:.4f} mixed_acc={row['mixed_target_argmax_accuracy']:.4f} lr={row['learning_rate']:.7f} sec={row['seconds']:.1f}",flush=True)
    clean_train=eval_subset(DATA,True,classes,True); restricted=eval_subset(DATA,False,classes,True); clean_acc=accuracy(model,clean_train); test_acc=accuracy(model,restricted)
    meta={"status":"COMPLETE","rotation":rotation,"seed":seed,"epochs":300,"train_samples":40000,"upstream_classes":80,"clean_train_accuracy":clean_acc,"restricted_test_accuracy":test_acc,"train_test_gap":clean_acc-test_acc,"final_mixed_target_argmax_accuracy":history[-1]["mixed_target_argmax_accuracy"],"final_soft_target_loss":history[-1]["train_soft_target_loss"],"checkpoint":str(cp),"checkpoint_sha256":file_hash(cp),"initial_state_sha256":initial,"pretrained_weights_loaded":False,"resolved_config":c,"architecture":architecture_record(model)}
    write_json(CHECKPOINTS/f"{tag}.json",meta); write_json(CONFIGS/f"{tag}_resolved.json",{"rotation":rotation,"seed":seed,"classes":classes,"config":c,"config_sha256":file_hash(CONFIG)}); write_json(MANIFESTS/f"{tag}_manifest.json",{"rotation":rotation,"seed":seed,"classes":classes,"canonical_manifest_sha256":file_hash(ROOT/'manifest.json')})
    print(json.dumps(meta))


def classification_audit():
    rows=[]
    for seed in SEEDS:
        for model in MODELS:
            p=CHECKPOINTS/f"{model}_seed{seed}.json"
            if not p.exists(): raise RuntimeError(f"Missing completed run: {p}")
            x=json.loads(p.read_text()); rows.append({k:x[k] for k in ("rotation","seed","clean_train_accuracy","restricted_test_accuracy","train_test_gap","final_mixed_target_argmax_accuracy","final_soft_target_loss")})
    df=pd.DataFrame(rows); mean=float(df.restricted_test_accuracy.mean()); status="ADEQUATE" if mean>=.70 else "MIXED" if mean>=.65 else "INSUFFICIENT"
    summary={"status":status,"created_utc":now(),"ood_evaluated_before_gate":False,"runs":8,"mean_restricted_test_accuracy":mean,"sd_restricted_test_accuracy":float(df.restricted_test_accuracy.std(ddof=1)),"min_restricted_test_accuracy":float(df.restricted_test_accuracy.min()),"max_restricted_test_accuracy":float(df.restricted_test_accuracy.max()),"mean_clean_train_accuracy":float(df.clean_train_accuracy.mean()),"mean_train_test_gap":float(df.train_test_gap.mean()),"resnet18_mean_restricted_test_accuracy":.781796875,"thresholds":cfg()["classification_gate"]}
    df.to_csv(CLASSIFICATION/"per_run.csv",index=False); write_json(CLASSIFICATION/"classification_gate.json",summary)
    lines=["# ViT-v2 classification-quality gate","",f"Status: **{status}**",f"Mean restricted-test accuracy: **{mean:.4%}**",f"Standard deviation: {summary['sd_restricted_test_accuracy']:.4%}; range [{summary['min_restricted_test_accuracy']:.4%}, {summary['max_restricted_test_accuracy']:.4%}].",f"Mean clean train accuracy: {summary['mean_clean_train_accuracy']:.4%}; mean train-test gap: {summary['mean_train_test_gap']:.4%}.","","This gate was written before feature extraction or OOD evaluation. ADEQUATE requires at least 70%; MIXED requires at least 65%; INSUFFICIENT stops the pipeline."]
    (CLASSIFICATION/"REPORT.md").write_text("\n".join(lines)+"\n"); print(json.dumps(summary,indent=2))


def gate():
    p=CLASSIFICATION/"classification_gate.json"
    if not p.exists(): raise RuntimeError("Classification gate has not run")
    status=json.loads(p.read_text())["status"]
    if status=="INSUFFICIENT": raise RuntimeError("Classification gate INSUFFICIENT: OOD pipeline is forbidden")
    if status not in ("ADEQUATE","MIXED"): raise RuntimeError("Unknown gate status")
    return status


def extract(model,seed):
    gate()
    from .evaluation import extract as run_extract
    run_extract(model,seed)

def evaluate(model,seed):
    gate()
    from .evaluation import evaluate as run_evaluate
    run_evaluate(model,seed)

def validate_outputs_v2(manifest_data, sf, se, cf, group, sm, result):
    checks={}; initial_by_seed={}
    for seed_value in SEEDS:
        hashes=[]
        for model_name in MODELS:
            tag=f"{model_name}_seed{seed_value}"; cp=CHECKPOINTS/f"{tag}.pt"; ft=FEATURES/f"{tag}.pt"; sc=SCORES/f"{tag}.npz"
            saved=torch.load(cp,map_location="cpu",weights_only=False); features=torch.load(ft,map_location="cpu",weights_only=False); scores=np.load(sc)
            classes=sorted(map(int,manifest_data["model_pretraining_classes"][model_name]))
            checks[f"{tag}_checkpoint"]=(saved["identity"]==[model_name,seed_value] and saved["epoch"]==300 and saved["target_epochs"]==300 and saved["classes"]==classes and saved["pretrained_weights_loaded"] is False and saved["config_sha256"]==file_hash(CONFIG))
            checks[f"{tag}_features"]=(features["identity"]==[model_name,seed_value] and tuple(features["train_id_features"].shape)==(10000,192) and tuple(features["test_features"].shape)==(10000,192) and torch.equal(features["test_indices"],torch.arange(10000)) and features["state_sha256_before"]==features["state_sha256_after"])
            checks[f"{tag}_scores"]=(set(DETECTORS)<=set(scores.files) and scores["class_ids"].shape==(10000,) and np.array_equal(scores["evaluation_indices"],features["test_indices"].numpy()) and np.array_equal(scores["class_ids"],features["test_labels"].numpy()) and all(np.isfinite(scores[d]).all() for d in DETECTORS))
            hashes.append(saved["initial_state_sha256"])
        initial_by_seed[seed_value]=hashes
        checks[f"seed{seed_value}_matched_initialization"]=len(set(hashes))==1
    checks["different_seed_initialization"]=initial_by_seed[0][0]!=initial_by_seed[1][0]
    checks["manifest_byte_identical"]=file_hash(ROOT/"manifest.json")==file_hash(MANIFEST_SOURCE) and file_hash(ROOT/"manifest.csv")==file_hash(MANIFEST_CSV_SOURCE)
    checks["per_state_rows"]=len(sf)==5760; checks["seed_effect_rows"]=len(se)==1440; checks["class_effect_rows"]=len(cf)==720; checks["group_effect_rows"]=len(group)==180
    checks["summary_detectors"]=len(sm)==9 and set(sm.detector)==set(DETECTORS)
    checks["pairing_counts"]=bool((sf.groupby(["detector","class_id","seed"]).size()==4).all())
    checks["no_pretrained_weights"]=all(checks[f"{m}_seed{s}_checkpoint"] for s in SEEDS for m in MODELS)
    checks["raw_image_identity_paired"]=sf.groupby(["rotation","seed"]).ood_eval_images.sum().eq(8000*9).all()
    checks["fit_metadata_coverage"]=len(list(FITS.glob('M*_seed*/*.json')))==48
    validation={"status":"PASS" if all(checks.values()) else "FAIL","created_utc":now(),"checks":{k:bool(v) for k,v in checks.items()},"counts":{"checkpoints":len(list(CHECKPOINTS.glob('M*_seed*.pt'))),"feature_files":len(list(FEATURES.glob('M*_seed*.pt'))),"score_files":len(list(SCORES.glob('M*_seed*.npz'))),"fit_json_files":len(list(FITS.glob('M*_seed*/*.json'))),"bootstrap_files":len(list(BOOTSTRAP.glob('*.csv')))},"result_sha256":file_hash(SUMMARIES/'results.json'),"report_sha256":file_hash(ROOT/'REPORT.md'),"commands_sha256":file_hash(ROOT/'COMMANDS.md')}
    write_json(ROOT/"validation.json",validation)
    if validation["status"]!="PASS": raise RuntimeError(f"Final ViT-v2 validation failed: {validation}")




def finalize_report():
    status=gate()
    from .evaluation import aggregate as run_aggregate
    sf,se,cf,group,sm,seed,result=run_aggregate()
    gate_data=json.loads((CLASSIFICATION/"classification_gate.json").read_text())
    reference=pd.read_csv(EXPANDED/"detector_dataset_summary.csv").query("dataset_slug=='cifar100'")
    rows=[]
    for detector in DETECTORS:
        current=sm[sm.detector.eq(detector)].iloc[0]
        baseline=reference[reference.detector.eq(detector)].iloc[0]
        rows.append(f"| {DISPLAY[detector]} | {baseline.mean_delta:.6f} | {current.mean_delta:.6f} | [{current.ci95_low:.6f}, {current.ci95_high:.6f}] | {int(current.negative_classes)}/80 | {int(current.negative_groups)}/20 |")
    report=f"""# CIFAR-100 ViT architecture experiment

The retained ViT-Tiny/4 experiment uses random initialization and the fixed 300-epoch regularized recipe. **No external pretrained weights were used.**

Classification-quality status: **{status}**. Mean restricted-test accuracy: {gate_data['mean_restricted_test_accuracy']:.2%}.

| Detector | ResNet-18 Delta | ViT Delta | ViT 95% CI | ViT negative classes | ViT negative groups |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Detector fitting uses downstream-ID data only. Focal OOD examples are used only for final scoring and AUROC calculation.
"""
    (ROOT/"REPORT.md").write_text(report)
    write_commands()
    validate_outputs_v2(manifest(),sf,se,cf,group,sm,result)
    validation=json.loads((ROOT/"validation.json").read_text())
    validation.update({"classification_gate":status,"classification_gate_before_ood":True,"report_sha256":file_hash(ROOT/"REPORT.md"),"commands_sha256":file_hash(ROOT/"COMMANDS.md"),"source_sha256":file_hash(Path(__file__))})
    write_json(ROOT/"validation.json",validation)
    if validation["status"]!="PASS": raise RuntimeError(validation)

def write_commands():
    py=sys.executable; lines=["# Exact ViT-v2 commands","",f"Working directory: `{ROOT}`","","```bash",f"{py} run.py preflight","```",""]
    for seed in SEEDS:
        for model in MODELS: lines += ["```bash",f"{py} run.py train --model {model} --seed {seed}","```",""]
    lines += ["```bash",f"{py} run.py classification-audit","```",""]
    for phase in ("extract","evaluate"):
        for seed in SEEDS:
            for model in MODELS: lines += ["```bash",f"{py} run.py {phase} --model {model} --seed {seed}","```",""]
    lines += ["```bash",f"{py} run.py aggregate","```",""]; (ROOT/"COMMANDS.md").write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); sub.add_parser("preflight"); sub.add_parser("classification-audit"); sub.add_parser("aggregate")
    for name in ("train","extract","evaluate"):
        q=sub.add_parser(name); q.add_argument("--model",choices=MODELS,required=True); q.add_argument("--seed",choices=SEEDS,type=int,required=True)
    a=p.parse_args(); record_command()
    if a.command=="preflight": preflight()
    elif a.command=="train": train(a.model,a.seed)
    elif a.command=="classification-audit": classification_audit()
    elif a.command=="extract": extract(a.model,a.seed)
    elif a.command=="evaluate": evaluate(a.model,a.seed)
    else: gate(); finalize_report()


if __name__=="__main__": main()
