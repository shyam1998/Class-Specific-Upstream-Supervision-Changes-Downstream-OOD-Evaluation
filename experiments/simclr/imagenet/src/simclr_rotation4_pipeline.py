from __future__ import annotations

import copy,json,math,os,random,time
from datetime import datetime,timezone
from pathlib import Path

from .data import RemappedImageFolder,transforms
from .rotation4_design import MODELS,flat_classes,load_json,model_wnids,sha256_file,verify_rotation_manifest
from .simclr_rotation4 import SimCLRModel,TwoViewTransform,initialized_states,make_backbone,nt_xent,simclr_transform,state_dict_sha256
from .verify import atomic_torch_save,safe_torch_load


def now():return datetime.now(timezone.utc).isoformat()


def paths(project: Path,config: dict):return {k:project/v for k,v in config["paths"].items() if k!="namespace"}


def verify_guard(project: Path,guard_path: Path):
    guard=load_json(guard_path)
    for relative,record in guard["artifacts"].items():
        target=project/relative
        if not target.is_file() or target.stat().st_size!=record["size_bytes"] or sha256_file(target)!=record["sha256"]:
            raise RuntimeError(f"Protected supervised artifact changed: {target}")
    return guard


def frozen(project: Path,config: dict):
    p=paths(project,config);manifest_path=project/config["source"]["rotation_manifest"]
    manifest=load_json(manifest_path);verify_rotation_manifest(manifest)
    mh=sha256_file(manifest_path);ph=sha256_file(project/"config.yaml")
    root=os.environ.get("IMAGENET_ROOT",config["dataset"]["root"])
    prereg={"source":{"rotation_manifest_sha256":mh},
      "dataset":{"root":root,"train_dir":config["dataset"]["train_dir"],"val_dir":config["dataset"]["val_dir"]},
      "simclr":config["simclr"],"downstream":config["downstream"]}
    memory={"status":"PASS","selected_mode":{"activation_checkpointing":False}}
    for key in ("checkpoints","features","artifacts","logs"):p[key].mkdir(parents=True,exist_ok=True)
    if not p["run_manifest"].is_file():p["run_manifest"].write_text('{"stages":{},"runs":{}}\n')
    return manifest,prereg,mh,ph,p,memory


def metadata(kind,model,seed,mh,ph,**extra):return {"kind":kind,"rotation_model":model,"condition":model,"seed":seed,
    "rotation_manifest_sha256":mh,"manifest_sha256":mh,"preregistration_sha256":ph,"namespace":"simclr_rotation4_v1",**extra}


def update_stage(path,stage,status,**extra):
    payload=load_json(path);payload.setdefault("stages",{})[stage]={"status":status,"timestamp_utc":now(),**extra}
    temp=path.with_suffix(".json.partial");temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temp,path)


def update_run(path,model,seed,**extra):
    payload=load_json(path);payload.setdefault("runs",{}).setdefault(f"{model}_seed{seed}",{}).update(extra)
    temp=path.with_suffix(".json.partial");temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temp,path)


def initial_path(p,seed,mh,ph):return p["checkpoints"]/f"initial_seed{seed}_p{ph[:10]}_m{mh[:10]}.pt"


def initial_state(project,config,seed):
    manifest,_,mh,ph,p,_=frozen(project,config);path=initial_path(p,seed,mh,ph)
    expected={"kind":"simclr_initialization","seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if path.exists():
        payload=safe_torch_load(path,expected)
        if state_dict_sha256(payload["encoder_state"])!=payload["metadata"]["encoder_sha256"] or state_dict_sha256(payload["projector_state"])!=payload["metadata"]["projector_sha256"]:
            raise RuntimeError("Canonical SimCLR initialization content hash mismatch")
        return payload,path
    encoder,projector=initialized_states(seed);eh=state_dict_sha256(encoder);jh=state_dict_sha256(projector)
    payload={"metadata":metadata("simclr_initialization","ALL_M1_M2_M3_M4",seed,mh,ph,encoder_sha256=eh,projector_sha256=jh),
      "encoder_state":encoder,"projector_state":projector};atomic_torch_save(payload,path);return payload,path


def checkpoint_path(p,model,seed,epoch):return p["checkpoints"]/f"{model}_seed{seed}"/f"epoch_{epoch:03d}.pt"


def train_upstream(project: Path,config: dict,model_id: str,seed: int):
    import torch
    from torch.utils.data import DataLoader
    manifest,prereg,mh,ph,p,memory=frozen(project,config);recipe=prereg["simclr"]
    final=checkpoint_path(p,model_id,seed,recipe["epochs"]);expected={"kind":"simclr_checkpoint","rotation_model":model_id,"seed":seed,
      "epoch":recipe["epochs"],"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if final.exists():safe_torch_load(final,expected);return final
    class_order=model_wnids(manifest,model_id);mapping={w:i for i,w in enumerate(class_order)}
    dataset=RemappedImageFolder.build(Path(prereg["dataset"]["root"])/prereg["dataset"]["train_dir"],mapping,TwoViewTransform(simclr_transform(config)))
    if set(dataset.targets)!=set(range(80)):raise RuntimeError(f"{model_id} missing a scientific upstream image class")
    generator=torch.Generator().manual_seed(seed);loader=DataLoader(dataset,batch_size=recipe["source_batch_size"],shuffle=True,
      num_workers=recipe["num_workers"],pin_memory=True,persistent_workers=True,drop_last=True,generator=generator)
    initial,initial_file=initial_state(project,config,seed);checkpointed=bool(memory["selected_mode"]["activation_checkpointing"])
    model=SimCLRModel(checkpointed).module;model.encoder.load_state_dict(initial["encoder_state"]);model.projector.load_state_dict(initial["projector_state"])
    if state_dict_sha256(model.encoder.state_dict())!=initial["metadata"]["encoder_sha256"] or state_dict_sha256(model.projector.state_dict())!=initial["metadata"]["projector_sha256"]:
        raise RuntimeError("Loaded scientific initialization mismatch")
    model.cuda().to(memory_format=torch.channels_last);optimizer=torch.optim.SGD(model.parameters(),lr=recipe["initial_lr"],momentum=recipe["momentum"],weight_decay=recipe["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=recipe["scheduler_t_max"],eta_min=recipe["scheduler_eta_min"])
    scaler=torch.amp.GradScaler("cuda",enabled=recipe["amp"]);run_dir=final.parent;run_dir.mkdir(parents=True,exist_ok=True)
    log_path=p["logs"]/f"{model_id}_seed{seed}.jsonl";log_path.parent.mkdir(parents=True,exist_ok=True)
    start_epoch=0;global_step=0;elapsed_before=0.0
    for candidate in sorted(run_dir.glob("epoch_*.pt"),reverse=True):
        try:
            payload=safe_torch_load(candidate,{"kind":"simclr_checkpoint","rotation_model":model_id,"seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph})
            if payload["metadata"]["epoch"]>=recipe["epochs"]:continue
            model.encoder.load_state_dict(payload["encoder_state"]);model.projector.load_state_dict(payload["projector_state"])
            optimizer.load_state_dict(payload["optimizer_state"]);scheduler.load_state_dict(payload["scheduler_state"]);scaler.load_state_dict(payload["scaler_state"])
            generator.set_state(payload["data_generator_state"]);start_epoch=int(payload["metadata"]["epoch"]);global_step=int(payload["global_step"]);elapsed_before=float(payload["metadata"].get("runtime_seconds",0));break
        except Exception as exc:print(f"Ignoring invalid SimCLR resume candidate {candidate}: {exc}",flush=True)
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);started=time.time()
    update_run(p["run_manifest"],model_id,seed,status="RUNNING",started_utc=now(),start_epoch=start_epoch,class_count=80,
      training_examples=len(dataset),batches_per_epoch=len(loader),encoder_initial_sha256=initial["metadata"]["encoder_sha256"],
      projector_initial_sha256=initial["metadata"]["projector_sha256"],initial_checkpoint=str(initial_file))
    for epoch in range(start_epoch,recipe["epochs"]):
        model.train();epoch_started=time.time();loss_sum=0.0;used_lr=optimizer.param_groups[0]["lr"]
        for (view1,view2),_selection_labels in loader:
            # Selection labels are deliberately discarded; loss accepts views only.
            if len(view1)!=recipe["source_batch_size"]:raise RuntimeError("Scientific contrastive batch is not exactly 256")
            view1=view1.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            view2=view2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=recipe["amp"]):
                _,z1=model(view1);_,z2=model(view2);loss,targets=nt_xent(z1,z2,recipe["temperature"])
            if len(targets)!=2*recipe["source_batch_size"] or not torch.isfinite(loss):raise RuntimeError("Invalid scientific NT-Xent loss/batch")
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach());global_step+=1
        scheduler.step();completed=epoch+1;runtime=elapsed_before+time.time()-started
        record={"timestamp_utc":now(),"model":model_id,"seed":seed,"epoch":completed,"epochs":recipe["epochs"],
          "mean_nt_xent_loss":loss_sum/len(loader),"lr_used":used_lr,"lr_after_epoch":optimizer.param_groups[0]["lr"],
          "epoch_seconds":time.time()-epoch_started,"runtime_seconds":runtime,"global_step":global_step,"source_batch_size":256,"representations_per_loss":512}
        with log_path.open("a") as handle:handle.write(json.dumps(record,sort_keys=True)+"\n")
        print(json.dumps(record,sort_keys=True),flush=True)
        if completed%recipe["checkpoint_every"]==0 or completed==recipe["epochs"]:
            encoder_hash=state_dict_sha256(model.encoder.state_dict());projector_hash=state_dict_sha256(model.projector.state_dict())
            meta=metadata("simclr_checkpoint",model_id,seed,mh,ph,epoch=completed,class_order=class_order,
              source_batch_size=256,representations_per_loss=512,temperature=.5,activation_checkpointing=checkpointed,channels_last=True,
              encoder_initial_sha256=initial["metadata"]["encoder_sha256"],projector_initial_sha256=initial["metadata"]["projector_sha256"],
              encoder_sha256=encoder_hash,projector_sha256=projector_hash,runtime_seconds=runtime,dry_run=False)
            atomic_torch_save({"metadata":meta,"encoder_state":model.encoder.state_dict(),"projector_state":model.projector.state_dict(),
              "optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict(),
              "global_step":global_step,"mean_epoch_loss":record["mean_nt_xent_loss"],"data_generator_state":generator.get_state()},checkpoint_path(p,model_id,seed,completed))
    update_run(p["run_manifest"],model_id,seed,status="COMPLETE",completed_utc=now(),runtime_seconds=runtime,global_step=global_step,
      final_checkpoint=str(final),final_encoder_sha256=encoder_hash,final_projector_sha256=projector_hash)
    return final


def feature_path(p,model,seed,split,mh,ph):return p["features"]/f"{model}_seed{seed}_{split}_p{ph[:10]}_m{mh[:10]}.pt"


def load_feature(path,expected):
    sidecar=path.with_suffix(".sha256")
    if not sidecar.exists() or sha256_file(path)!=sidecar.read_text().split()[0]:raise RuntimeError(f"Feature cache hash mismatch: {path}")
    return safe_torch_load(path,expected)


def extract_features(project,config,model_id,seed,split):
    import torch
    from torch.utils.data import DataLoader
    manifest,prereg,mh,ph,p,_=frozen(project,config);output=feature_path(p,model_id,seed,split,mh,ph)
    expected={"kind":"simclr_feature_cache","rotation_model":model_id,"seed":seed,"split":split,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if output.exists():load_feature(output,expected);return output
    rows=flat_classes(manifest);roles={"d"} if split=="train" else {"d","a","b","r1","r2"};wnids=sorted(x["wnid"] for x in rows if x["role"] in roles);mapping={w:i for i,w in enumerate(wnids)}
    dataset=RemappedImageFolder.build(Path(prereg["dataset"]["root"])/prereg["dataset"][f"{split}_dir"],mapping,transforms(False))
    loader=DataLoader(dataset,batch_size=config["downstream"]["feature_batch_size"],shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    checkpoint=safe_torch_load(checkpoint_path(p,model_id,seed,prereg["simclr"]["epochs"]),{"kind":"simclr_checkpoint","rotation_model":model_id,"seed":seed,
      "epoch":prereg["simclr"]["epochs"],"rotation_manifest_sha256":mh,"preregistration_sha256":ph})
    encoder=make_backbone();encoder.load_state_dict(checkpoint["encoder_state"]);before=state_dict_sha256(encoder.state_dict());encoder.cuda().eval()
    features=[];labels=[]
    with torch.inference_mode():
        for images,target in loader:
            value=encoder(images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            if tuple(value.shape[1:])!=(2048,) or not torch.isfinite(value).all():raise RuntimeError("Invalid SimCLR downstream features")
            features.append(value.cpu());labels.append(target.cpu())
    after=state_dict_sha256(encoder.state_dict())
    if before!=after:raise RuntimeError("Encoder/BN state mutated during frozen extraction")
    sample_ids=[str(Path(path).relative_to(Path(prereg["dataset"]["root"]))) for path,_ in dataset.samples]
    meta=metadata("simclr_feature_cache",model_id,seed,mh,ph,split=split,feature_dim=2048,normalized=False,wnid_order=wnids,
      sample_count=len(dataset),sample_ids=sample_ids,encoder_sha256=before,source_checkpoint=str(checkpoint_path(p,model_id,seed,200)))
    atomic_torch_save({"metadata":meta,"features":torch.cat(features),"labels":torch.cat(labels)},output);output.with_suffix(".sha256").write_text(f"{sha256_file(output)}  {output.name}\n");return output


def probe_path(p,model,seed,mh,ph):return p["checkpoints"]/f"probe_{model}_seed{seed}_p{ph[:10]}_m{mh[:10]}.pt"


def train_probe(project,config,model_id,seed):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    manifest,prereg,mh,ph,p,_=frozen(project,config);output=probe_path(p,model_id,seed,mh,ph)
    expected={"kind":"simclr_linear_probe","rotation_model":model_id,"seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if output.exists():safe_torch_load(output,expected);return output
    train=load_feature(feature_path(p,model_id,seed,"train",mh,ph),{"kind":"simclr_feature_cache","rotation_model":model_id,"seed":seed,"split":"train","rotation_manifest_sha256":mh,"preregistration_sha256":ph})
    val=load_feature(feature_path(p,model_id,seed,"val",mh,ph),{"kind":"simclr_feature_cache","rotation_model":model_id,"seed":seed,"split":"val","rotation_manifest_sha256":mh,"preregistration_sha256":ph})
    d=sorted(x["wnid"] for x in flat_classes(manifest) if x["role"]=="d")
    if train["metadata"]["wnid_order"]!=d or set(train["labels"].tolist())!=set(range(20)):raise RuntimeError("Probe train cache is not exactly d classes")
    recipe=prereg["downstream"]["probe"];torch.manual_seed(10_000+seed);torch.cuda.manual_seed_all(10_000+seed);probe=torch.nn.Linear(2048,20).cuda();initial_hash=state_dict_sha256(probe.state_dict())
    loader=DataLoader(TensorDataset(train["features"],train["labels"]),batch_size=recipe["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(10_000+seed))
    optimizer=torch.optim.SGD(probe.parameters(),lr=recipe["initial_lr"],momentum=recipe["momentum"],weight_decay=recipe["weight_decay"])
    for epoch in range(recipe["epochs"]):
        probe.train();lr=recipe["initial_lr"]*.5*(1+math.cos(math.pi*epoch/recipe["epochs"]))
        for group in optimizer.param_groups:group["lr"]=lr
        for features,targets in loader:
            optimizer.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(probe(features.cuda()),targets.cuda()).backward();optimizer.step()
    probe.eval();val_wnids=[val["metadata"]["wnid_order"][index] for index in val["labels"].tolist()];mask=torch.tensor([w in set(d) for w in val_wnids]);targets=torch.tensor([d.index(w) for w in val_wnids if w in set(d)])
    with torch.inference_mode():predicted=probe(val["features"][mask].cuda()).argmax(1).cpu()
    accuracy=float((predicted==targets).float().mean());meta=metadata("simclr_linear_probe",model_id,seed,mh,ph,input_dim=2048,output_dim=20,
      epochs=recipe["epochs"],initialization_seed=10_000+seed,initial_state_sha256=initial_hash,downstream_id_val_accuracy=accuracy)
    atomic_torch_save({"metadata":meta,"model_state":probe.cpu().state_dict()},output);update_run(p["run_manifest"],model_id,seed,probe_accuracy=accuracy,probe_checkpoint=str(output));return output
