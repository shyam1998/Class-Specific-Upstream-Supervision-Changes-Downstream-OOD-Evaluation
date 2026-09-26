from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_ood import auroc,logit_scores
from .rotation4_analysis import knn_scores_cuda
from .rotation4_design import MODELS,flat_classes,sha256_file
from .simclr_rotation4_pipeline import feature_path,frozen,load_feature,now,probe_path,update_stage,verify_guard
from .verify import safe_torch_load

DETECTORS=("knn","energy","msp")


def evaluate_models(project,config):
    import torch
    manifest,prereg,mh,ph,p,_=frozen(project,config);rows=flat_classes(manifest);id_wnids={x["wnid"] for x in rows if x["role"]=="d"};candidates=[x for x in rows if x["role"]!="d"]
    output=[];identity=None;probe_hashes={}
    for seed in prereg["design"]["seeds"]:
        probe_hashes[seed]=[]
        for model_id in MODELS:
            train=load_feature(feature_path(p,model_id,seed,"train",mh,ph),{"kind":"simclr_feature_cache","rotation_model":model_id,"seed":seed,"split":"train","rotation_manifest_sha256":mh,"preregistration_sha256":ph})
            val=load_feature(feature_path(p,model_id,seed,"val",mh,ph),{"kind":"simclr_feature_cache","rotation_model":model_id,"seed":seed,"split":"val","rotation_manifest_sha256":mh,"preregistration_sha256":ph})
            probe=safe_torch_load(probe_path(p,model_id,seed,mh,ph),{"kind":"simclr_linear_probe","rotation_model":model_id,"seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph})
            probe_hashes[seed].append(probe["metadata"]["initial_state_sha256"])
            signature=(tuple(val["metadata"]["wnid_order"]),tuple(val["metadata"]["sample_ids"]),tuple(val["labels"].tolist()))
            if identity is None:identity=signature
            elif signature!=identity:raise RuntimeError("Downstream validation image identities differ across provenance states")
            label_wnids=np.asarray(val["metadata"]["wnid_order"])[val["labels"].numpy()];id_mask=np.isin(label_wnids,list(id_wnids))
            if id_mask.sum()!=1000:raise RuntimeError("Expected 1000 downstream-ID validation images")
            knn=knn_scores_cuda(train["features"].cuda(),val["features"].cuda(),k=50).cpu().numpy();energy,msp=logit_scores(val["features"],probe["model_state"])
            scores={"knn":knn,"energy":energy.numpy(),"msp":msp.numpy()};centroids=torch.stack([train["features"][train["labels"]==index].mean(0) for index in range(20)])
            for item in candidates:
                ood_mask=label_wnids==item["wnid"]
                if ood_mask.sum()!=50:raise RuntimeError(f"Expected 50 validation images for {item['wnid']}")
                record={"rotation_model":model_id,"seed":seed,"group_id":item["group_id"],"semantic_parent":item["semantic_parent"],"role":item["role"],"slot":item["slot"],
                  "wnid":item["wnid"],"class_name":item["class_name"],"presence_state":"withheld" if item["withheld_model"]==model_id else "present"}
                for detector in DETECTORS:record[f"auroc_{detector}"]=auroc(scores[detector][id_mask],scores[detector][ood_mask])
                record["nearest_id_centroid_distance"]=float(torch.cdist(val["features"][torch.from_numpy(ood_mask)].float(),centroids.float()).min(1).values.mean())
                output.append(record)
        if len(set(probe_hashes[seed]))!=1:raise RuntimeError(f"Probe initialization mismatch within seed {seed}")
    raw=pd.DataFrame(output)
    if len(raw)!=640:raise RuntimeError(f"Expected 640 model/class AUROC rows, got {len(raw)}")
    raw.to_csv(p["artifacts"]/"model_class_aurocs.csv",index=False);return raw


def build_metrics(raw,manifest):
    items={x["wnid"]:x for x in flat_classes(manifest) if x["role"]!="d"};records=[]
    for seed in sorted(raw.seed.unique()):
        for wnid,item in sorted(items.items(),key=lambda pair:(pair[1]["group_id"],pair[1]["slot"])):
            subset=raw[(raw.seed==seed)&(raw.wnid==wnid)].set_index("rotation_model");withheld=item["withheld_model"];present=item["supervised_models"]
            if set(subset.index)!=set(MODELS) or subset.loc[withheld,"presence_state"]!="withheld" or any(subset.loc[m,"presence_state"]!="present" for m in present):raise RuntimeError("Presence mapping mismatch")
            r={"group_id":item["group_id"],"semantic_parent":item["semantic_parent"],"role":item["role"],"slot":item["slot"],"wnid":wnid,"class_name":item["class_name"],
              "seed":seed,"withheld_model":withheld,"present_models":";".join(present)}
            for detector in DETECTORS:
                for m in MODELS:r[f"auroc_{detector}_{m}"]=float(subset.loc[m,f"auroc_{detector}"])
                values=np.asarray([subset.loc[m,f"auroc_{detector}"] for m in present],float);r[f"auroc_{detector}_withheld"]=float(subset.loc[withheld,f"auroc_{detector}"])
                for index,value in enumerate(values,1):r[f"auroc_{detector}_present_{index}"]=float(value)
                r[f"auroc_{detector}_present_mean"]=float(values.mean());r[f"auroc_{detector}_present_sd"]=float(values.std(ddof=1));r[f"delta_ssl_{detector}"]=r[f"auroc_{detector}_withheld"]-r[f"auroc_{detector}_present_mean"]
            geometry=np.asarray([subset.loc[m,"nearest_id_centroid_distance"] for m in present],float);r["geometry_withheld"]=float(subset.loc[withheld,"nearest_id_centroid_distance"]);r["geometry_present_mean"]=float(geometry.mean());r["delta_geometry"]=r["geometry_withheld"]-r["geometry_present_mean"]
            r["gap_withheld"]=r["auroc_knn_withheld"]-r["auroc_energy_withheld"];r["gap_present_mean"]=r["auroc_knn_present_mean"]-r["auroc_energy_present_mean"]
            r["delta_ssl_gap"]=r["gap_withheld"]-r["gap_present_mean"];r["reversal"]=r["gap_withheld"]*r["gap_present_mean"]<0
            r["ordering_withheld"]=">".join(sorted(DETECTORS,key=lambda d:r[f"auroc_{d}_withheld"],reverse=True));r["ordering_present_mean"]=">".join(sorted(DETECTORS,key=lambda d:r[f"auroc_{d}_present_mean"],reverse=True));records.append(r)
    frame=pd.DataFrame(records)
    if len(frame)!=160 or not np.allclose(frame.delta_ssl_gap,frame.delta_ssl_knn-frame.delta_ssl_energy,atol=1e-12,rtol=0):raise RuntimeError("Class×seed estimand invariant failed")
    if not all(frame[c].between(0,1).all() for c in frame if c.startswith("auroc_")):raise RuntimeError("AUROC outside [0,1]")
    return frame


def average_classes(frame):
    keys=["group_id","semantic_parent","role","slot","wnid","class_name","withheld_model","present_models"]
    numeric=[c for c in frame if c not in keys+["seed","reversal","ordering_withheld","ordering_present_mean"]]
    result=frame.groupby(keys,as_index=False)[numeric].mean();result["reversal"]=(result.gap_withheld*result.gap_present_mean)<0
    result["ordering_withheld"]=[">".join(sorted(DETECTORS,key=lambda d:row[f"auroc_{d}_withheld"],reverse=True)) for _,row in result.iterrows()]
    result["ordering_present_mean"]=[">".join(sorted(DETECTORS,key=lambda d:row[f"auroc_{d}_present_mean"],reverse=True)) for _,row in result.iterrows()]
    if len(result)!=80:raise RuntimeError("Expected 80 seed-averaged candidates")
    return result


def group_bootstrap(frame,column,draws,seed):
    arrays=[frame[frame.group_id==g][column].to_numpy() for g in sorted(frame.group_id.unique())]
    if len(arrays)!=20 or any(len(x)!=4 for x in arrays):raise RuntimeError("Bootstrap requires 20 groups x four candidates")
    values=np.stack(arrays);rng=np.random.default_rng(seed);idx=rng.integers(0,20,size=(draws,20));means=values[idx].reshape(draws,80).mean(1);low,high=np.quantile(means,[.025,.975]);raw=frame[column].to_numpy()
    return {"metric":column,"mean":float(raw.mean()),"median":float(np.median(raw)),"sd":float(raw.std(ddof=1)),"min":float(raw.min()),"max":float(raw.max()),
      "negative":int((raw<0).sum()),"zero":int((raw==0).sum()),"positive":int((raw>0).sum()),"ci95_low":float(low),"ci95_high":float(high),"draws":draws,"seed":seed,"n_groups":20,"n_classes":80}


def inventories(p):
    import torch
    checkpoints=[]
    for path in sorted(p["checkpoints"].glob("**/*.pt")):
        payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True);m=payload.get("metadata",{});checkpoints.append({"path":str(path),"size_bytes":path.stat().st_size,"kind":m.get("kind"),"model":m.get("rotation_model"),"seed":m.get("seed"),"epoch":m.get("epoch"),"encoder_sha256":m.get("encoder_sha256")})
    features=[{"path":str(path),"size_bytes":path.stat().st_size,"sha256":path.with_suffix(".sha256").read_text().split()[0]} for path in sorted(p["features"].glob("*.pt"))]
    (p["artifacts"]/"checkpoint_inventory.json").write_text(json.dumps(checkpoints,indent=2,sort_keys=True)+"\n");(p["artifacts"]/"feature_cache_inventory.json").write_text(json.dumps(features,indent=2,sort_keys=True)+"\n")


def analyze(project,config):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr,spearmanr
    manifest,prereg,mh,ph,p,memory=frozen(project,config);p["artifacts"].mkdir(parents=True,exist_ok=True)
    raw=evaluate_models(project,config);class_seed=build_metrics(raw,manifest);class_seed.to_csv(p["artifacts"]/"class_level_metrics.csv",index=False);classes=average_classes(class_seed);classes.to_csv(p["artifacts"]/"class_summary.csv",index=False)
    group=classes.groupby(["group_id","semantic_parent"],as_index=False).agg(mean_delta_ssl_knn=("delta_ssl_knn","mean"),mean_delta_ssl_energy=("delta_ssl_energy","mean"),mean_delta_ssl_msp=("delta_ssl_msp","mean"),mean_delta_ssl_gap=("delta_ssl_gap","mean"),mean_delta_geometry=("delta_geometry","mean"),negative_knn_classes=("delta_ssl_knn",lambda x:int((x<0).sum())),reversal_classes=("reversal","sum"));group.to_csv(p["artifacts"]/"group_level_summary.csv",index=False)
    seed=class_seed.groupby("seed",as_index=False).agg(mean_delta_ssl_knn=("delta_ssl_knn","mean"),negative_delta_ssl_knn=("delta_ssl_knn",lambda x:int((x<0).sum())),mean_delta_ssl_energy=("delta_ssl_energy","mean"),mean_delta_ssl_msp=("delta_ssl_msp","mean"),mean_delta_ssl_gap=("delta_ssl_gap","mean"),reversal_count=("reversal","sum"));seed.to_csv(p["artifacts"]/"seed_level_summary.csv",index=False)
    primary=group_bootstrap(classes,"delta_ssl_knn",prereg["bootstrap"]["draws"],prereg["bootstrap"]["primary_seed"]);primary["group_means"]=group[["group_id","semantic_parent","mean_delta_ssl_knn"]].to_dict("records");primary["seed_summaries"]=seed.to_dict("records")
    sim_pass=primary["mean"]<=-.03 and primary["ci95_high"]<0 and primary["negative"]>=60
    (p["artifacts"]/"bootstrap_summary.json").write_text(json.dumps({"primary":primary,"decision":"SIMCLR_NEGATIVE_EFFECT_REPLICATED" if sim_pass else "SIMCLR_NEGATIVE_EFFECT_NOT_REPLICATED"},indent=2,sort_keys=True)+"\n")
    supervised=pd.read_csv(project/"artifacts/rotation4_v1/class_summary.csv")[["group_id","wnid","class_name","delta_knn","delta_energy","delta_gap"]].rename(columns={"delta_knn":"delta_sup_knn","delta_energy":"delta_sup_energy","delta_gap":"delta_sup_gap"})
    paired=classes.merge(supervised,on=["group_id","wnid","class_name"],validate="one_to_one");paired["gamma"]=paired.delta_sup_knn-paired.delta_ssl_knn
    moderation=group_bootstrap(paired,"gamma",prereg["bootstrap"]["draws"],prereg["bootstrap"]["moderation_seed"]);gamma_group=paired.groupby("group_id",as_index=False).gamma.mean();moderation["group_means"]=gamma_group.to_dict("records");moderation["pearson_r"]=float(pearsonr(paired.delta_sup_knn,paired.delta_ssl_knn).statistic);moderation["spearman_r"]=float(spearmanr(paired.delta_sup_knn,paired.delta_ssl_knn).statistic)
    mod_pass=moderation["mean"]<=-.03 and moderation["ci95_high"]<0;moderation["decision"]="SUPERVISION_MODERATION_REPLICATED" if mod_pass else "SUPERVISION_MODERATION_NOT_REPLICATED";(p["artifacts"]/"moderation_summary.json").write_text(json.dumps(moderation,indent=2,sort_keys=True)+"\n")
    geometry={"mean_delta":float(classes.delta_geometry.mean()),"pearson_r":float(pearsonr(classes.delta_geometry,classes.delta_ssl_knn).statistic),"spearman_r":float(spearmanr(classes.delta_geometry,classes.delta_ssl_knn).statistic)}
    secondary={"delta_energy":group_bootstrap(classes,"delta_ssl_energy",10000,prereg["bootstrap"]["primary_seed"]),"delta_msp":group_bootstrap(classes,"delta_ssl_msp",10000,prereg["bootstrap"]["primary_seed"]),"delta_gap":group_bootstrap(classes,"delta_ssl_gap",10000,prereg["bootstrap"]["primary_seed"]),"class_averaged_reversals":int(classes.reversal.sum()),"class_seed_reversals":int(class_seed.reversal.sum()),"ordering_withheld":classes.ordering_withheld.value_counts().to_dict(),"ordering_present_mean":classes.ordering_present_mean.value_counts().to_dict(),"geometry":geometry}
    (p["artifacts"]/"secondary_summary.json").write_text(json.dumps(secondary,indent=2,sort_keys=True)+"\n")
    verdict={"simclr":"SIMCLR_NEGATIVE_EFFECT_REPLICATED" if sim_pass else "SIMCLR_NEGATIVE_EFFECT_NOT_REPLICATED","moderation":moderation["decision"]};(p["artifacts"]/"verdict.json").write_text(json.dumps(verdict,indent=2,sort_keys=True)+"\n");(p["artifacts"]/"verdict.txt").write_text(verdict["simclr"]+"\n"+verdict["moderation"]+"\n")
    for data,column,xlabel,name in ((classes,"delta_ssl_knn","Delta SSL kNN","delta_ssl_knn_distribution.png"),(classes,"delta_ssl_gap","Delta SSL (kNN-Energy)","delta_ssl_gap_distribution.png"),(paired,"gamma","Gamma = Delta supervised - Delta SimCLR","gamma_distribution.png")):
        fig,ax=plt.subplots(figsize=(8,4.8));ax.hist(data[column],bins=20,edgecolor="white");ax.axvline(0,color="black",lw=.8);ax.set_xlabel(xlabel);ax.set_ylabel("Classes");fig.tight_layout();fig.savefig(p["artifacts"]/name,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(paired.delta_ssl_knn,paired.delta_sup_knn);ax.axhline(0,color="black",lw=.7);ax.axvline(0,color="black",lw=.7);ax.set_xlabel("SimCLR Delta kNN");ax.set_ylabel("Supervised Delta kNN");fig.tight_layout();fig.savefig(p["artifacts"]/"supervised_vs_simclr_delta_knn.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(classes.delta_geometry,classes.delta_ssl_knn);ax.axhline(0,color="black",lw=.7);ax.axvline(0,color="black",lw=.7);ax.set_xlabel("Delta nearest-ID-centroid distance");ax.set_ylabel("Delta SSL kNN");fig.tight_layout();fig.savefig(p["artifacts"]/"geometry_vs_delta_ssl_knn.png",dpi=180);plt.close(fig)
    losses=[]
    for path in sorted(p["logs"].glob("*.jsonl")):
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()];losses.append({"run":path.stem,"epochs":len(records),"initial_loss":records[0]["mean_nt_xent_loss"],"final_loss":records[-1]["mean_nt_xent_loss"],"final_lr_after_epoch":records[-1]["lr_after_epoch"],"runtime_seconds":records[-1]["runtime_seconds"]})
    probes=[]
    for seed_value in prereg["design"]["seeds"]:
        for model_id in MODELS:
            payload=safe_torch_load(probe_path(p,model_id,seed_value,mh,ph),{"kind":"simclr_linear_probe","rotation_model":model_id,"seed":seed_value,"rotation_manifest_sha256":mh,"preregistration_sha256":ph});probes.append({"model":model_id,"seed":seed_value,"accuracy":payload["metadata"]["downstream_id_val_accuracy"]})
    verify_guard(project,p["source_guard"]);guard_recheck={"status":"PASS","timestamp_utc":now(),"protected_count":len(json.loads(p["source_guard"].read_text())["artifacts"])};inventories(p)
    report=["# ImageNet SimCLR grouped rotating leave-one-out experiment","","## A. Design","","The frozen supervised rotation4_v1 semantic membership and downstream task were reused exactly. WNIDs selected the unlabeled image pools but did not enter NT-Xent.","","## B. CIFAR-protocol-to-ImageNet mapping","",f"```json\n{json.dumps(prereg['simclr'],indent=2,sort_keys=True)}\n```","","## C. Allowed architecture/resolution adaptations","",f"```json\n{json.dumps(prereg['allowed_adaptations'],indent=2,sort_keys=True)}\n```","","## D. True-batch-256 hardware feasibility","",f"Standard AMP/channels-last OOMed; activation checkpointing passed with {memory['attempts'][-1]['peak_allocated_bytes']} peak allocated bytes. All 512 representations participated in one NT-Xent loss.","","## E. Dry-run invariants","",f"All {len(json.loads(p['dry_audit_json'].read_text())['checks'])} invariants passed.","","## F. SimCLR training sanity","",f"```json\n{json.dumps({'losses':losses,'probe_accuracies':probes},indent=2,sort_keys=True)}\n```","","## G. Primary Delta_SSL result","",f"Mean {primary['mean']:.6f}; median {primary['median']:.6f}; SD {primary['sd']:.6f}; range [{primary['min']:.6f}, {primary['max']:.6f}]; signs {primary['negative']}/{primary['zero']}/{primary['positive']} negative/zero/positive; 20-group 95% CI [{primary['ci95_low']:.6f}, {primary['ci95_high']:.6f}]. **{verdict['simclr']}**","","## H. Paired supervision moderation Gamma","",f"Mean {moderation['mean']:.6f}; median {moderation['median']:.6f}; signs {moderation['negative']}/{moderation['zero']}/{moderation['positive']}; CI [{moderation['ci95_low']:.6f}, {moderation['ci95_high']:.6f}]. **{verdict['moderation']}**","","## I. Energy / MSP / detector-gap secondaries","",f"```json\n{json.dumps(secondary,indent=2,sort_keys=True)}\n```","","## J. Geometry secondary","",f"```json\n{json.dumps(geometry,indent=2,sort_keys=True)}\n```","","## K. Limitations","","This compares supervised and SimCLR training pipelines, not a pure labels-only intervention. Activation checkpointing was required by the 20 GiB device. Geometry is descriptive and does not establish mediation.","","## L. Protected supervised artifact hash recheck","",f"```json\n{json.dumps(guard_recheck,indent=2,sort_keys=True)}\n```","","## M. Final pre-specified verdicts","",verdict["simclr"]+"  ",verdict["moderation"],"","No post-result tuning occurred. Null, positive, and contradictory class effects were retained."]
    (p["artifacts"]/"report.md").write_text("\n".join(report)+"\n");update_stage(p["run_manifest"],"analysis","COMPLETE",verdict=verdict,protected_recheck=guard_recheck);return verdict,primary,moderation
