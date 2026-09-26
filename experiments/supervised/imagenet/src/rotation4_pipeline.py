from __future__ import annotations

import copy,json,math,os,random,time
from datetime import datetime,timezone
from pathlib import Path

from .data import RemappedImageFolder,transforms
from .models import make_encoder,make_resnet50,state_dict_sha256
from .rotation4_design import MODELS,flat_classes,load_json,model_wnids,sha256_file,verify_rotation_manifest
from .verify import atomic_torch_save,safe_torch_load


def now():return datetime.now(timezone.utc).isoformat()


def paths(project: Path,config: dict):return {k:project/v for k,v in config["paths"].items() if k!="namespace"}


def frozen(project: Path,config: dict):
    p=paths(project,config);manifest=load_json(p["rotation_manifest_json"]);verify_rotation_manifest(manifest)
    mh=sha256_file(p["rotation_manifest_json"]);ph=sha256_file(project/"config.yaml")
    root=os.environ.get("IMAGENET_ROOT",config["dataset"]["root"])
    prereg={"rotation_manifest_sha256":mh,
      "dataset":{"root":root,"train_directory":config["dataset"]["train_dir"],"val_directory":config["dataset"]["val_dir"]},
      "upstream_training":config["upstream"],"linear_probe":config["downstream"]["probe"]}
    for key in ("checkpoints","features","logs","artifacts","reports"):p[key].mkdir(parents=True,exist_ok=True)
    if not p["run_manifest"].is_file():p["run_manifest"].write_text('{"stages":{},"runs":{}}\n')
    return manifest,prereg,mh,ph,p


def metadata(kind,model,seed,mh,ph,**extra):return {"kind":kind,"rotation_model":model,"condition":model,"seed":seed,
    "rotation_manifest_sha256":mh,"manifest_sha256":mh,"preregistration_sha256":ph,"namespace":"rotation4_v1",**extra}


def update_run_manifest(path: Path,stage: str,status: str,**extra):
    payload=json.loads(path.read_text());payload.setdefault("stages",{})[stage]={"status":status,"timestamp_utc":now(),**extra}
    temporary=path.with_suffix(".json.partial");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)


def update_run(path: Path,model: str,seed: int,**values):
    payload=json.loads(path.read_text());key=f"{model}_seed{seed}";payload.setdefault("runs",{}).setdefault(key,{}).update(values)
    temporary=path.with_suffix(".json.partial");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)


def initial_state(project: Path,config: dict,seed: int):
    import torch
    manifest,_,mh,ph,p=frozen(project,config);path=p["checkpoints"]/f"initial_seed{seed}_m{mh[:12]}.pt"
    expected={"kind":"initialization","seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if path.exists():
        payload=safe_torch_load(path,expected)
        if state_dict_sha256(payload["model_state"])!=payload["metadata"]["initial_state_sha256"]:raise RuntimeError("Initial-state content hash mismatch")
        return payload["model_state"],payload["metadata"]["initial_state_sha256"],path
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=make_resnet50(80);state=copy.deepcopy(model.state_dict());digest=state_dict_sha256(state)
    atomic_torch_save({"metadata":metadata("initialization","ALL_M1_M2_M3_M4",seed,mh,ph,initial_state_sha256=digest),"model_state":state},path)
    return state,digest,path


def lr_at(epoch_fraction,epochs,warmup,base_lr):
    if epoch_fraction<warmup:return base_lr*epoch_fraction/warmup
    progress=(epoch_fraction-warmup)/(epochs-warmup);return base_lr*.5*(1+math.cos(math.pi*min(progress,1.0)))


def checkpoint_path(p,model,seed,epoch):return p["checkpoints"]/f"{model}_seed{seed}"/f"epoch_{epoch:03d}.pt"


def train_upstream(project: Path,config: dict,model_id: str,seed: int):
    import torch
    from torch.utils.data import DataLoader
    if model_id not in MODELS:raise ValueError(model_id)
    if not torch.cuda.is_available():raise RuntimeError("Full rotation4 training requires CUDA")
    manifest,prereg,mh,ph,p=frozen(project,config)
    final=checkpoint_path(p,model_id,seed,100);expected={"kind":"upstream_checkpoint","rotation_model":model_id,"seed":seed,
        "epoch":100,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if final.exists():safe_torch_load(final,expected);return final
    class_order=model_wnids(manifest,model_id);mapping={w:i for i,w in enumerate(class_order)}
    dataset=RemappedImageFolder.build(Path(prereg["dataset"]["root"])/prereg["dataset"]["train_directory"],mapping,transforms(True))
    if set(dataset.targets)!=set(range(80)):raise RuntimeError(f"{model_id} has a missing training class")
    recipe=prereg["upstream_training"];generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=recipe["microbatch_size"],shuffle=True,num_workers=8,pin_memory=True,
        generator=generator,persistent_workers=True,drop_last=True)
    if recipe["microbatch_size"]*recipe["accumulation_steps"]!=recipe["effective_batch_size"]:raise RuntimeError("Effective batch mismatch")
    initial,initial_hash,initial_path=initial_state(project,config,seed);network=make_resnet50(80);network.load_state_dict(initial)
    if state_dict_sha256(network.state_dict())!=initial_hash:raise RuntimeError("Loaded initialization mismatch")
    device=torch.device("cuda");network.to(device);optimizer=torch.optim.SGD(network.parameters(),lr=recipe["initial_lr"],
        momentum=recipe["momentum"],weight_decay=recipe["weight_decay"]);scaler=torch.amp.GradScaler("cuda",enabled=recipe["amp"])
    run_dir=final.parent;run_dir.mkdir(parents=True,exist_ok=True);log_path=p["logs"]/f"{model_id}_seed{seed}.jsonl";log_path.parent.mkdir(parents=True,exist_ok=True)
    start_epoch=0;global_step=0;elapsed_before=0.0
    for candidate in sorted(run_dir.glob("epoch_*.pt"),reverse=True):
        try:
            resumed=safe_torch_load(candidate,{"kind":"upstream_checkpoint","rotation_model":model_id,"seed":seed,
                "rotation_manifest_sha256":mh,"preregistration_sha256":ph})
            epoch=int(resumed["metadata"]["epoch"])
            if epoch>=100:continue
            network.load_state_dict(resumed["model_state"]);optimizer.load_state_dict(resumed["optimizer_state"])
            scaler.load_state_dict(resumed["scaler_state"]);generator.set_state(resumed["data_generator_state"])
            start_epoch=epoch;global_step=int(resumed["global_step"]);elapsed_before=float(resumed["metadata"].get("runtime_seconds",0));break
        except Exception as exc:print(f"Ignoring invalid rotation4 resume candidate {candidate}: {exc}",flush=True)
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);started=time.time()
    update_run(p["run_manifest"],model_id,seed,status="RUNNING",started_utc=now(),start_epoch=start_epoch,
        class_count=80,training_examples=len(dataset),initial_state_sha256=initial_hash,initial_checkpoint=str(initial_path))
    for epoch in range(start_epoch,recipe["epochs"]):
        epoch_started=time.time();network.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.0
        for batch_index,(images,targets) in enumerate(loader):
            lr=lr_at(epoch+batch_index/max(1,len(loader)),recipe["epochs"],5,recipe["initial_lr"])
            for group in optimizer.param_groups:group["lr"]=lr
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=recipe["amp"]):
                loss=torch.nn.functional.cross_entropy(network(images),targets)/recipe["accumulation_steps"]
            scaler.scale(loss).backward();loss_sum+=float(loss)*recipe["accumulation_steps"]
            if (batch_index+1)%recipe["accumulation_steps"]==0 or batch_index+1==len(loader):
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);global_step+=1
        completed=epoch+1;runtime=elapsed_before+time.time()-started;record={"timestamp_utc":now(),"model":model_id,"seed":seed,
            "epoch":completed,"epochs":recipe["epochs"],"mean_loss":loss_sum/max(1,len(loader)),"lr":lr,
            "epoch_seconds":time.time()-epoch_started,"runtime_seconds":runtime,"global_step":global_step}
        with log_path.open("a") as handle:handle.write(json.dumps(record,sort_keys=True)+"\n")
        print(json.dumps(record,sort_keys=True),flush=True)
        if completed%recipe["checkpoint_every"]==0 or completed==recipe["epochs"]:
            target=checkpoint_path(p,model_id,seed,completed);meta=metadata("upstream_checkpoint",model_id,seed,mh,ph,
                epoch=completed,initial_state_sha256=initial_hash,output_dim=80,class_order=class_order,
                microbatch_size=recipe["microbatch_size"],accumulation_steps=recipe["accumulation_steps"],
                effective_batch_size=recipe["effective_batch_size"],initial_lr=recipe["initial_lr"],runtime_seconds=runtime,
                global_step=global_step,dry_run=False)
            atomic_torch_save({"metadata":meta,"model_state":network.state_dict(),"optimizer_state":optimizer.state_dict(),
                "scaler_state":scaler.state_dict(),"mean_epoch_loss":record["mean_loss"],"global_step":global_step,
                "data_generator_state":generator.get_state()},target)
    update_run(p["run_manifest"],model_id,seed,status="COMPLETE",completed_utc=now(),final_checkpoint=str(final),
        runtime_seconds=runtime,global_step=global_step)
    return final


def feature_path(p,model,seed,split,mh,ph):return p["features"]/f"{model}_seed{seed}_{split}_p{ph[:10]}_m{mh[:10]}.pt"


def load_feature(path,expected):
    sidecar=path.with_suffix(".sha256")
    if not sidecar.exists() or sha256_file(path)!=sidecar.read_text().split()[0]:raise RuntimeError(f"Invalid feature cache hash: {path}")
    return safe_torch_load(path,expected)


def extract_features(project: Path,config: dict,model_id: str,seed: int,split: str):
    import torch
    from torch.utils.data import DataLoader
    manifest,prereg,mh,ph,p=frozen(project,config);output=feature_path(p,model_id,seed,split,mh,ph)
    expected={"kind":"feature_cache","rotation_model":model_id,"seed":seed,"split":split,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if output.exists():load_feature(output,expected);return output
    rows=flat_classes(manifest);roles={"d"} if split=="train" else {"d","a","b","r1","r2"}
    wnids=sorted(x["wnid"] for x in rows if x["role"] in roles);mapping={w:i for i,w in enumerate(wnids)}
    dataset=RemappedImageFolder.build(Path(prereg["dataset"]["root"])/prereg["dataset"][f"{split}_directory"],mapping,transforms(False))
    loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    checkpoint=checkpoint_path(p,model_id,seed,100);payload=safe_torch_load(checkpoint,{"kind":"upstream_checkpoint",
        "rotation_model":model_id,"seed":seed,"epoch":100,"rotation_manifest_sha256":mh,"preregistration_sha256":ph})
    network=make_resnet50(80);network.load_state_dict(payload["model_state"]);encoder=make_encoder(network).cuda().eval()
    features=[];labels=[]
    with torch.inference_mode():
        for images,target in loader:
            value=encoder(images.cuda(non_blocking=True));
            if value.ndim!=2 or value.shape[1]!=2048:raise RuntimeError("Bad rotation4 feature shape")
            features.append(value.cpu());labels.append(target.cpu())
    meta=metadata("feature_cache",model_id,seed,mh,ph,split=split,feature_dim=2048,normalized=False,
        wnid_order=wnids,sample_count=len(dataset),source_checkpoint=str(checkpoint),source_checkpoint_size=checkpoint.stat().st_size)
    atomic_torch_save({"metadata":meta,"features":torch.cat(features),"labels":torch.cat(labels)},output)
    output.with_suffix(".sha256").write_text(f"{sha256_file(output)}  {output.name}\n");return output


def probe_path(p,model,seed,mh,ph):return p["checkpoints"]/f"probe_{model}_seed{seed}_p{ph[:10]}_m{mh[:10]}.pt"


def train_probe(project: Path,config: dict,model_id: str,seed: int):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    manifest,prereg,mh,ph,p=frozen(project,config);output=probe_path(p,model_id,seed,mh,ph)
    expected={"kind":"linear_probe","rotation_model":model_id,"seed":seed,"rotation_manifest_sha256":mh,"preregistration_sha256":ph}
    if output.exists():safe_torch_load(output,expected);return output
    cache=load_feature(feature_path(p,model_id,seed,"train",mh,ph),{"kind":"feature_cache","rotation_model":model_id,
        "seed":seed,"split":"train","rotation_manifest_sha256":mh,"preregistration_sha256":ph})
    d=sorted(x["wnid"] for x in flat_classes(manifest) if x["role"]=="d")
    if cache["metadata"]["wnid_order"]!=d:raise RuntimeError("Probe train cache is not the fixed 20 d classes")
    recipe=prereg["linear_probe"];torch.manual_seed(10_000+seed);torch.cuda.manual_seed_all(10_000+seed)
    probe=torch.nn.Linear(2048,20).cuda();initial_hash=state_dict_sha256(probe.state_dict())
    loader=DataLoader(TensorDataset(cache["features"],cache["labels"]),batch_size=recipe["batch_size"],shuffle=True,
        generator=torch.Generator().manual_seed(10_000+seed))
    optimizer=torch.optim.SGD(probe.parameters(),lr=recipe["initial_lr"],momentum=recipe["momentum"],weight_decay=0)
    for epoch in range(recipe["epochs"]):
        probe.train();lr=recipe["initial_lr"]*.5*(1+math.cos(math.pi*epoch/recipe["epochs"]))
        for group in optimizer.param_groups:group["lr"]=lr
        for features,targets in loader:
            optimizer.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(probe(features.cuda()),targets.cuda()).backward();optimizer.step()
    atomic_torch_save({"metadata":metadata("linear_probe",model_id,seed,mh,ph,epochs=recipe["epochs"],input_dim=2048,
        output_dim=20,initial_state_sha256=initial_hash,initialization_seed=10_000+seed),"model_state":probe.cpu().state_dict()},output)
    return output
