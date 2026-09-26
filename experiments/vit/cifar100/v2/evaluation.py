from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from . import run as ctx
from analysis.detectors.src.detectors import (fit_gradorth,fit_mahalanobis,fit_neco,fit_vim,probe_logits,score_gradorth,score_mahalanobis,score_nci,score_neco,score_vim,state_hash as detector_state_hash)
ROOT=ctx.ROOT; DATA=ctx.DATA; CHECKPOINTS=ctx.CHECKPOINTS; FEATURES=ctx.FEATURES; PROBES=ctx.PROBES; SCORES=ctx.SCORES; FITS=ctx.FITS; SUMMARIES=ctx.SUMMARIES; BOOTSTRAP=ctx.BOOTSTRAP; EXPANDED=ctx.EXPANDED
MODELS=ctx.MODELS; SEEDS=ctx.SEEDS; DETECTORS=ctx.DETECTORS; DISPLAY=ctx.DISPLAY
CIFAR100_STD=ctx.CIFAR100_STD; CifarViTTinyV2=ctx.CifarViTTinyV2; init_model=ctx.init_model; state_hash=ctx.state_hash; file_hash=ctx.file_hash; write_json=ctx.write_json; sha_array=ctx.sha_array; sha256_array=ctx.sha_array; eval_subset=ctx.eval_subset; seed_everything=ctx.seed_everything; cfg=ctx.cfg; manifest=ctx.manifest; now=ctx.now

class EncoderProbe(nn.Module):
 def __init__(self,encoder,probe):super().__init__();self.encoder=encoder;self.probe=probe
 def features(self,x):return self.encoder.features(x)
 def forward(self,x):return self.probe(self.features(x))
class IndexedDataset(Dataset):
 def __init__(self,dataset):self.dataset=dataset
 def __len__(self):return len(self.dataset)
 def __getitem__(self,i):
  x,y=self.dataset[i];return x,y,i
def candidate_map(value):
 out={}
 for g in value["groups"]:
  for role in ("c1","c2","c3","c4"):
   c=g[role];out[int(c["fine_id"])]={"group_id":int(g["coarse_id"]),"group_name":g["coarse_name"],"class_id":int(c["fine_id"]),"class_name":c["fine_name"],"role":role,"original_role":c["original_role"],"withheld_model":c["withheld_model"]}
 return out
def fit_nci_vit(reference,alpha=.01):
 x=np.asarray(reference,dtype=np.float64);arrays={"global_mean":x.mean(0),"alpha":np.asarray(alpha,dtype=np.float64)};meta={"detector":"nci","fit_dtype":"float64","feature_dim":int(x.shape[1]),"reference_count":len(x),"alpha":alpha,"norm_filter":"L1","alpha_selection":"fixed before evaluation; no sweep or OOD data"};meta["fit_state_sha256"]=detector_state_hash(arrays,meta);return arrays,meta

def collect(model: CifarViTTinyV2, dataset, device="cuda"):
    loader=DataLoader(dataset,batch_size=cfg()["feature_batch_size"],shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    feats=[]; logits=[]; labels=[]
    model.eval(); model.requires_grad_(False)
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device,non_blocking=True); f=model.features(x); feats.append(f.cpu()); logits.append(model.classifier(f).cpu()); labels.append(y)
    return torch.cat(feats),torch.cat(logits),torch.cat(labels)


def extract(rotation: str, seed: int) -> None:
    cfg=cfg(); manifest=manifest(); tag=f"{rotation}_seed{seed}"; cp=CHECKPOINTS/f"{tag}.pt"
    s=torch.load(cp,map_location="cpu",weights_only=False); model,_=init_model(seed); model.load_state_dict(s["model_state"],strict=True); before=state_hash(model.state_dict()); model=model.cuda()
    d=set(map(int,manifest["downstream_id_classes"])); train_ds=eval_subset(DATA,True,d,remap=False); test_ds=eval_subset(DATA,False,set(range(100)),remap=False)
    train_f,train_up,train_y=collect(model,train_ds); test_f,test_up,test_y=collect(model,test_ds); after=state_hash(model.cpu().state_dict())
    if before!=after or train_f.shape!=(10000,192) or test_f.shape!=(10000,192): raise RuntimeError("Feature extraction invariant failed")
    out=FEATURES/f"{tag}.pt"
    torch.save({"identity":[rotation,seed],"train_id_features":train_f,"train_id_labels":train_y,"train_id_indices":torch.tensor(train_ds.indices),"train_id_upstream_logits":train_up,"test_features":test_f,"test_labels":test_y,"test_indices":torch.tensor(test_ds.indices),"test_upstream_logits":test_up,"feature_interface":"final LayerNorm CLS token before upstream classifier","feature_dim":192,"encoder_eval":True,"parameters_frozen":True,"state_sha256_before":before,"state_sha256_after":after,"checkpoint_sha256":file_hash(cp)},out)
    np.savez_compressed(FEATURES/f"{tag}_upstream_head.npz",weight=model.classifier.weight.detach().numpy(),bias=model.classifier.bias.detach().numpy(),classes=np.asarray(s["classes"]))
    write_json(FEATURES/f"{tag}.json",{"status":"PASS","path":str(out),"sha256":file_hash(out),"train_id_shape":list(train_f.shape),"test_shape":list(test_f.shape),"upstream_logits_shape":list(test_up.shape),"image_identity_hashes":{"train":sha256_array(np.asarray(train_ds.indices,dtype=np.int64)),"test":sha256_array(np.asarray(test_ds.indices,dtype=np.int64))}})
    print(json.dumps({"status":"PASS","features":str(out)}))


def train_probe(rotation: str, seed: int, data: dict, manifest: dict) -> tuple[nn.Linear,dict]:
    tag=f"{rotation}_seed{seed}"; path=PROBES/f"{tag}.pt"; d=sorted(map(int,manifest["downstream_id_classes"])); mapping={v:i for i,v in enumerate(d)}
    x=data["train_id_features"].float(); y=torch.tensor([mapping[int(v)] for v in data["train_id_labels"]])
    seed_everything(seed,True); gen=torch.Generator().manual_seed(seed); head=nn.Linear(192,20); opt=SGD(head.parameters(),lr=.1,momentum=.9); sched=CosineAnnealingLR(opt,T_max=50); start=0; history=[]
    if path.exists():
        s=torch.load(path,map_location="cpu",weights_only=False)
        if s["identity"]!=[rotation,seed] or s["feature_sha256"]!=file_hash(FEATURES/f"{tag}.pt"): raise RuntimeError("Probe cache mismatch")
        head.load_state_dict(s["head"]); opt.load_state_dict(s["optimizer"]); sched.load_state_dict(s["scheduler"]); start=s["epoch"]; history=s["history"]
    loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True,generator=gen,num_workers=0)
    for e in range(start,50):
        total=0.; n=0
        for f,t in loader:
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(head(f),t); loss.backward(); opt.step(); total+=float(loss)*len(t); n+=len(t)
        sched.step(); history.append({"epoch":e+1,"loss":total/n})
        torch.save({"identity":[rotation,seed],"epoch":e+1,"head":head.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"history":history,"d_classes":d,"feature_sha256":file_hash(FEATURES/f"{tag}.pt"),"recipe":cfg()["probe_recipe"]},path)
    test_mask=np.isin(data["test_labels"].numpy(),d); test_y=torch.tensor([mapping[int(v)] for v in data["test_labels"].numpy()[test_mask]])
    head.eval(); acc=float((head(data["test_features"][test_mask].float()).argmax(1)==test_y).float().mean())
    meta={"status":"PASS","rotation":rotation,"seed":seed,"checkpoint":str(path),"sha256":file_hash(path),"downstream_id_test_accuracy":acc,"head_shape":[20,192],"recipe":cfg()["probe_recipe"]}; write_json(PROBES/f"{tag}.json",meta)
    return head,meta


def knn_scores(query: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    ref=F.normalize(reference.float(),dim=1).cuda(); out=[]
    with torch.inference_mode():
        for i in range(0,len(query),512):
            q=F.normalize(query[i:i+512].float(),dim=1).cuda(); dist=1-q@ref.T; out.append(dist.topk(50,largest=False,dim=1).values.mean(1).cpu())
    return torch.cat(out).numpy()


def save_fit(tag: str, detector: str, arrays: dict, meta: dict, adaptation: dict) -> None:
    d=FITS/tag; d.mkdir(parents=True,exist_ok=True); np.savez_compressed(d/f"{detector}.npz",**arrays)
    write_json(d/f"{detector}.json",{**meta,"status":"PASS","rotation":tag.split('_')[0],"seed":int(tag[-1]),"score_orientation":"higher_is_more_ood","fit_data":"fixed downstream-ID training features only","focal_ood_used_for_fit_or_tuning":False,"architecture_adaptation":adaptation,"fit_file":str(d/f'{detector}.npz'),"fit_file_sha256":file_hash(d/f'{detector}.npz')})


def odin_batch(model, images, temperature, epsilon, std):
    images=images.detach().requires_grad_(True); logits=model(images); pred=logits.detach().argmax(1); loss=F.cross_entropy(logits/temperature,pred,reduction="sum"); grad=torch.autograd.grad(loss,images,only_inputs=True)[0]; signed=torch.where(grad>=0,torch.ones_like(grad),-torch.ones_like(grad)); perturbed=images.detach()-epsilon*signed/std
    with torch.inference_mode(): score=1-torch.softmax(model(perturbed)/temperature,1).max(1).values
    return score,logits.detach(),grad.detach(),perturbed.detach()


def odin_scores(rotation: str, seed: int, probe: nn.Linear, cached: dict) -> tuple[np.ndarray,dict]:
    cfg=cfg(); tag=f"{rotation}_seed{seed}"; s=torch.load(CHECKPOINTS/f"{tag}.pt",map_location="cpu",weights_only=False); vit,_=init_model(seed); vit.load_state_dict(s["model_state"],strict=True); model=EncoderProbe(vit,probe).cuda().eval(); model.requires_grad_(False)
    before=state_hash(model.state_dict()); ds=IndexedDataset(eval_subset(DATA,False,set(range(100)),remap=False)); loader=DataLoader(ds,batch_size=cfg["odin"]["batch_size"],shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True); std=torch.tensor(CIFAR100_STD,device="cuda").view(1,3,1,1); out=[]; positions=[]; audit=None
    for bi,(images,labels,pos) in enumerate(loader):
        images=images.cuda(non_blocking=True)
        if bi==0:
            with torch.inference_mode(): raw=model.features(images)
            expected=cached["test_features"][pos].cuda(); match={"max_abs":float((raw-expected).abs().max()),"min_cosine":float(F.cosine_similarity(raw,expected,dim=1).min())}
        score,logits,grad,pert=odin_batch(model,images,1000.,.002,std); out.append(score.cpu().numpy()); positions+=pos.tolist()
        if bi==0:
            step=torch.mean(torch.abs((pert-images)*std),dim=(0,2,3)).cpu().numpy(); audit={"raw_feature_match":match,"gradient_finite":bool(torch.isfinite(grad).all()),"score_finite":bool(torch.isfinite(score).all()),"raw_pixel_step_by_channel":step.tolist(),"max_step_error":float(np.max(np.abs(step-.002)))}
    values=np.concatenate(out).astype(np.float32); after=state_hash(model.cpu().state_dict())
    if positions!=list(range(10000)) or before!=after or not np.isfinite(values).all() or audit["raw_feature_match"]["min_cosine"]<.99999 or audit["max_step_error"]>1e-6: raise RuntimeError("ODIN audit failure")
    return values,{"status":"PASS","temperature":1000.,"epsilon_raw_pixel_units":.002,"batch_size":512,"precision":"FP32","clipping":False,"parameters_frozen":True,"state_immutable":before==after,"first_batch":audit,"focal_ood_used_for_tuning":False}


def evaluate(rotation: str, seed: int) -> None:
    manifest=manifest(); tag=f"{rotation}_seed{seed}"; data=torch.load(FEATURES/f"{tag}.pt",map_location="cpu",weights_only=False); probe,pmeta=train_probe(rotation,seed,data,manifest)
    ref=data["train_id_features"].numpy(); labels=data["train_id_labels"].numpy().astype(str); ev=data["test_features"].numpy(); w=probe.weight.detach().numpy(); b=probe.bias.detach().numpy()
    logits=probe_logits(ev,w,b)
    logits_t=torch.from_numpy(logits)
    scores={"knn":knn_scores(data["test_features"],data["train_id_features"]),"energy":(-torch.logsumexp(logits_t,dim=1)).numpy(),"msp":(1-logits_t.softmax(1).max(1).values).numpy()}
    st,meta=fit_mahalanobis(ref,labels); scores["mahalanobis"]=score_mahalanobis(ev,st); save_fit(tag,"mahalanobis",st,meta,{"change":"feature dimension inferred as 192; final CLS features replace ResNet pooled features","id_only":True})
    st,meta=fit_vim(ref,w,b); scores["vim"]=score_vim(ev,w,b,st); save_fit(tag,"vim",st,meta,{"change":"192D CLS feature and 20x192 probe weights; canonical dimension rule gives principal dimension 96","id_only":True})
    st,meta=fit_neco(ref); meta["architecture_branch"]="ViT CLS final feature"; meta["fit_state_sha256"]=detector_state_hash(st,{k:v for k,v in meta.items() if k!='fit_state_sha256'}); scores["neco"]=score_neco(ev,st); save_fit(tag,"neco",st,meta,{"change":"standalone canonical NECO applied to final 192D CLS features; same 90% ID explained-variance rule","id_only":True})
    st,meta=fit_nci_vit(ref,.01); scores["nci"]=score_nci(ev,w,b,st); save_fit(tag,"nci",st,meta,{"change":"fixed canonical CIFAR alpha=0.01 applied at 192D because canonical function only enumerates 512/2048; no sweep","id_only":True})
    st,meta=fit_gradorth(ref); scores["gradorth"]=score_gradorth(ev,w,b,st); save_fit(tag,"gradorth",st,meta,{"change":"192D CLS features and 20x192 downstream probe; same full-ID uncentered 97% SVD rule","id_only":True})
    scores["odin"],odin_meta=odin_scores(rotation,seed,probe,data); write_json(FITS/tag/"odin.json",{**odin_meta,"architecture_adaptation":{"change":"input gradient passes through ViT CLS encoder and unchanged downstream probe; epsilon/T unchanged","id_only_settings":True}})
    if set(scores)!=set(DETECTORS) or any(len(x)!=10000 or not np.isfinite(x).all() for x in scores.values()): raise RuntimeError("Detector score coverage failure")
    out=SCORES/f"{tag}.npz"; np.savez_compressed(out,evaluation_indices=data["test_indices"].numpy(),class_ids=data["test_labels"].numpy(),downstream_logits=logits,**scores)
    write_json(SCORES/f"{tag}.json",{"status":"PASS","rotation":rotation,"seed":seed,"detectors":list(DETECTORS),"score_orientation":"higher_is_more_ood","evaluation_images":10000,"id_reference_images":10000,"file":str(out),"sha256":file_hash(out),"probe":pmeta,"focal_ood_used_for_fit_or_tuning":False})
    print(json.dumps({"status":"PASS","scores":str(out),"detectors":len(scores)}))


def auc(id_score,ood_score):
    return float(roc_auc_score(np.r_[np.zeros(len(id_score)),np.ones(len(ood_score))],np.r_[id_score,ood_score]))


def aggregate() -> None:
    manifest=manifest(); cmap=candidate_map(manifest); d=np.asarray(manifest["downstream_id_classes"]); state=[]
    for rotation in MODELS:
        for seed in SEEDS:
            z=np.load(SCORES/f"{rotation}_seed{seed}.npz"); y=z["class_ids"]; idm=np.isin(y,d)
            if idm.sum()!=2000: raise RuntimeError("ID test count mismatch")
            for cid,item in cmap.items():
                om=y==cid
                for detector in DETECTORS:
                    state.append({"dataset":"CIFAR-100","architecture":"ViT-Tiny/4","detector":detector,**item,"seed":seed,"rotation":rotation,"state":"withheld" if rotation==item["withheld_model"] else "present","id_eval_images":2000,"ood_eval_images":100,"auroc":auc(z[detector][idm],z[detector][om])})
    sf=pd.DataFrame(state); sf.to_csv(SUMMARIES/"per_state_aurocs.csv",index=False)
    seed_rows=[]
    for keys,g in sf.groupby(["detector","group_id","group_name","class_id","class_name","role","original_role","withheld_model","seed"]):
        detector,gid,gname,cid,cname,role,orole,wm,seed=keys; wh=g[g.state.eq('withheld')]; pr=g[g.state.eq('present')]
        if len(wh)!=1 or len(pr)!=3: raise RuntimeError("Paired state aggregation failure")
        by=g.set_index('rotation').auroc.to_dict(); aw=float(wh.auroc.iloc[0]); ap=float(pr.auroc.mean())
        seed_rows.append({"dataset":"CIFAR-100","architecture":"ViT-Tiny/4","detector":detector,"group_id":gid,"group_name":gname,"class_id":cid,"class_name":cname,"role":role,"original_role":orole,"withheld_model":wm,"seed":seed,"auroc_withheld":aw,"auroc_present_mean":ap,"delta":aw-ap,**{f"auroc_{m}":by[m] for m in MODELS}})
    se=pd.DataFrame(seed_rows); se.to_csv(SUMMARIES/"seed_level_effects.csv",index=False)
    class_rows=[]
    ids=["detector","group_id","group_name","class_id","class_name","role","original_role","withheld_model"]
    for keys,g in se.groupby(ids):
        base=dict(zip(ids,keys)); class_rows.append({"dataset":"CIFAR-100","architecture":"ViT-Tiny/4",**base,"delta_seed0":float(g[g.seed.eq(0)].delta.iloc[0]),"delta_seed1":float(g[g.seed.eq(1)].delta.iloc[0]),"delta":float(g.delta.mean()),"mean_auroc_withheld":float(g.auroc_withheld.mean()),"mean_auroc_present":float(g.auroc_present_mean.mean())})
    cf=pd.DataFrame(class_rows); cf.to_csv(SUMMARIES/"class_level_effects.csv",index=False)
    group=cf.groupby(["detector","group_id","group_name"],as_index=False).agg(classes=("class_id","size"),mean_delta=("delta","mean")); group.to_csv(SUMMARIES/"group_level_effects.csv",index=False)
    summary=[]; seed_summary=[]
    rng_seed=cfg()["bootstrap_seed"]
    for detector in DETECTORS:
        x=cf[cf.detector.eq(detector)].sort_values('class_id'); gv=group[group.detector.eq(detector)].sort_values('group_id').mean_delta.to_numpy(); vals=x.delta.to_numpy(); rng=np.random.default_rng(rng_seed); draws=gv[rng.integers(0,20,size=(10000,20))].mean(1); pd.DataFrame({"draw":np.arange(1,10001),"mean_delta":draws}).to_csv(BOOTSTRAP/f"{detector}.csv",index=False)
        summary.append({"detector":detector,"detector_display":DISPLAY[detector],"classes":80,"groups":20,"mean_delta":float(vals.mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"negative_classes":int((vals<0).sum()),"positive_classes":int((vals>0).sum()),"zero_classes":int((vals==0).sum()),"negative_groups":int((gv<0).sum()),"positive_groups":int((gv>0).sum()),"zero_groups":int((gv==0).sum()),"min_delta":float(vals.min()),"q1_delta":float(np.quantile(vals,.25)),"median_delta":float(np.median(vals)),"q3_delta":float(np.quantile(vals,.75)),"max_delta":float(vals.max()),"mean_auroc_withheld":float(x.mean_auroc_withheld.mean()),"mean_auroc_supervised":float(x.mean_auroc_present.mean()),"bootstrap_draws":10000,"bootstrap_seed":rng_seed})
        for seed in SEEDS:
            v=se[(se.detector.eq(detector))&(se.seed.eq(seed))].delta.to_numpy(); seed_summary.append({"detector":detector,"seed":seed,"classes":80,"mean_delta":float(v.mean()),"negative_classes":int((v<0).sum()),"positive_classes":int((v>0).sum()),"median_delta":float(np.median(v))})
    sm=pd.DataFrame(summary); sm.to_csv(SUMMARIES/"detector_summary.csv",index=False); pd.DataFrame(seed_summary).to_csv(SUMMARIES/"seed_specific_summary.csv",index=False)
    # Exact canonical ranking-reversal definition: average state-specific detector gaps over seeds, then compare signs.
    reversal=[]
    for cid,item in cmap.items():
        q=se[se.class_id.eq(cid)]; gaps={}
        for state_name,col in (("withheld","auroc_withheld"),("present","auroc_present_mean")):
            k=q[q.detector.eq('knn')][col].mean(); e=q[q.detector.eq('energy')][col].mean(); gaps[state_name]=float(k-e)
        reversal.append({**item,"gap_withheld":gaps["withheld"],"gap_present":gaps["present"],"reversal":gaps["withheld"]*gaps["present"]<0,"tie":gaps["withheld"]==0 or gaps["present"]==0})
    rv=pd.DataFrame(reversal); rv.to_csv(SUMMARIES/"knn_energy_ranking_reversals.csv",index=False)
    wide=cf.pivot(index=["group_id","group_name","class_id","class_name"],columns="detector",values="delta").reset_index(); signs=np.sign(wide[list(DETECTORS)].to_numpy()); wide["distinct_signs"]=[len(set(row)) for row in signs]; wide["sign_disagreement"]=wide.distinct_signs.gt(1); wide.to_csv(SUMMARIES/"cross_detector_sign_disagreement.csv",index=False)
    reference=pd.read_csv(EXPANDED/"detector_dataset_summary.csv").query("dataset_slug=='cifar100'")
    comp=sm.merge(reference[["detector","mean_delta","ci95_low","ci95_high","negative_classes","negative_groups"]],on="detector",suffixes=("_vit","_resnet")); comp.to_csv(SUMMARIES/"resnet18_vs_vit.csv",index=False)
    result={"status":"PASS","created_utc":now(),"vit_knn_energy_reversals":int(rv.reversal.sum()),"vit_knn_energy_ties":int(rv.tie.sum()),"resnet18_knn_energy_reversals":24,"vit_cross_detector_sign_disagreement":int(wide.sign_disagreement.sum()),"resnet18_cross_detector_sign_disagreement":61,"detectors":summary}
    write_json(SUMMARIES/"results.json",result)
    print(json.dumps(result,indent=2))
    return sf,se,cf,group,sm,pd.DataFrame(seed_summary),result

