#!/usr/bin/env python3
"""iNaturalist 2021 FULL-native ViT-Tiny/16 provenance experiment."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW,SGD
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
from torch.utils.data import DataLoader,Dataset,Subset,TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.ops import stochastic_depth
from torchvision.transforms import InterpolationMode,v2

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
SOURCE=Path(os.environ.get('INAT_SUPERVISED_ROOT',REPO/'outputs/supervised/inaturalist')); EXPANDED=Path(os.environ.get('DETECTOR_RESULTS_ROOT',REPO/'outputs/detectors'))
CONFIG=ROOT/'config.json'; DATA_DEFAULT=REPO/'data/inaturalist'
MANIFEST=ROOT/'manifests/frozen_manifest.csv'; FULLCSV=ROOT/'manifests/train_full_native_selected.csv'; MINICSV=ROOT/'manifests/train_mini_selected.csv'; EVALCSV=ROOT/'manifests/val_selected.csv'
CHECKPOINTS=ROOT/'checkpoints'; LOGS=ROOT/'logs'; TRAINING=ROOT/'training_metrics'; CLASSIFICATION=ROOT/'classification_results'; FEATURES=ROOT/'features'; PROBES=ROOT/'probes'; SCORES=ROOT/'detector_outputs'; FITS=ROOT/'detector_fit_states'; BOOTSTRAP=ROOT/'bootstrap_samples'; SUMMARIES=ROOT/'summaries'
MODELS=('M1','M2','M3','M4'); SEEDS=(0,1); DETECTORS=('knn','energy','msp','mahalanobis','vim','neco','nci','gradorth','odin')
DISPLAY={'knn':'kNN','energy':'Energy','msp':'MSP','mahalanobis':'Mahalanobis','vim':'ViM','neco':'NECO','nci':'NCI','gradorth':'GradOrth','odin':'ODIN'}
sys.path.insert(0,str(REPO))
from analysis.detectors.src.detectors import probe_logits,fit_mahalanobis,score_mahalanobis,fit_vim,score_vim,fit_neco,score_neco,score_nci,fit_gradorth,score_gradorth,state_hash as detector_state_hash

def cfg(): return json.loads(CONFIG.read_text())
def data_root(): return Path(os.environ.get('INAT_ROOT',cfg()['data_root']))
def now(): return datetime.now(timezone.utc).isoformat()
def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.partial'); q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(q,p)
def file_hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def state_digest(s):
    h=hashlib.sha256()
    for k in sorted(s):
        x=s[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(x.dtype).encode());h.update(np.asarray(x.shape,dtype=np.int64).tobytes());h.update(x.numpy().tobytes())
    return h.hexdigest()
def ensure():
    for p in (CHECKPOINTS,LOGS,TRAINING,CLASSIFICATION,FEATURES,PROBES,SCORES,FITS,BOOTSTRAP,SUMMARIES,ROOT/'smoke'):p.mkdir(parents=True,exist_ok=True)
def record():
    ensure()
    with (LOGS/'commands.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'cwd':str(Path.cwd()),'argv':[sys.executable,*sys.argv]})+'\n')

def manifest_rows(): return read_csv(MANIFEST)
def rotation_rows(model): return sorted(read_csv(ROOT/f'manifests/rotation_{model}.csv'),key=lambda r:int(r['local_training_label']))
def class_order(model): return [int(r['category_id']) for r in rotation_rows(model)]
def d_classes(): return [int(x['category_id']) for x in manifest_rows() if x['role']=='d']
def candidates():
    return {int(x['category_id']):{'group_id':x['group_id'],'group_name':x['parent_name'],'class_id':int(x['category_id']),'class_name':x['scientific_name'],'role':x['role'],'withheld_model':x['withheld_model']} for x in manifest_rows() if x['role']!='d'}

class ViTTiny16(nn.Module):
    def __init__(self,n=80,drop=.1):
        super().__init__();self.vit=VisionTransformer(image_size=224,patch_size=16,num_layers=12,num_heads=3,hidden_dim=192,mlp_dim=768,dropout=0.,attention_dropout=0.,num_classes=n);self.drop=float(drop)
    @property
    def classifier(self):return self.vit.heads.head
    def features(self,x):
        x=self.vit._process_input(x);x=torch.cat([self.vit.class_token.expand(len(x),-1,-1),x],1);x=x+self.vit.encoder.pos_embedding;x=self.vit.encoder.dropout(x);layers=list(self.vit.encoder.layers)
        for i,b in enumerate(layers):
            p=self.drop*i/max(1,len(layers)-1);y=b.ln_1(x);y,_=b.self_attention(y,y,y,need_weights=False);x=x+stochastic_depth(b.dropout(y),p,'row',self.training);x=x+stochastic_depth(b.mlp(b.ln_2(x)),p,'row',self.training)
        return self.vit.encoder.ln(x)[:,0]
    def forward(self,x):return self.classifier(self.features(x))
class EncoderProbe(nn.Module):
    def __init__(self,e,p):super().__init__();self.encoder=e;self.probe=p
    def features(self,x):return self.encoder.features(x)
    def forward(self,x):return self.probe(self.features(x))

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;g=torch.Generator();g.manual_seed(s);return g
def init_model(s):g=seed_all(s);return ViTTiny16(80,cfg()['augmentation']['drop_path_rate']),g
def eval_tf():return v2.Compose([v2.Resize(256,interpolation=InterpolationMode.BICUBIC),v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(cfg()['normalization_mean'],cfg()['normalization_std'])])
def train_tf():return v2.Compose([v2.RandomResizedCrop(224,interpolation=InterpolationMode.BICUBIC),v2.RandomHorizontalFlip(),v2.RandAugment(2,9,31,InterpolationMode.BICUBIC,0),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(cfg()['normalization_mean'],cfg()['normalization_std']),v2.RandomErasing(.25,scale=(.02,.33),ratio=(.3,3.3),value=0.)])
def mixer():return v2.RandomChoice([v2.MixUp(alpha=.8,num_classes=80),v2.CutMix(alpha=1.,num_classes=80)],p=[.5,.5])
class IndexedImages(Dataset):
    def __init__(self,rows,transform,label_map=None):self.rows=list(rows);self.tf=transform;self.label_map=label_map
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(data_root()/r['file_name']) as im:x=self.tf(im.convert('RGB'))
        cid=int(r['category_id']);y=self.label_map[cid] if self.label_map is not None else cid
        return x,y,int(r['image_id']),i

def sorted_rows(table):return sorted(read_csv(table),key=lambda r:(int(r['category_id']),int(r['image_id'])))
def training_dataset(model,augment=True):
    order=class_order(model);label={c:i for i,c in enumerate(order)};rows=[r for r in sorted_rows(FULLCSV) if int(r['category_id']) in label]
    expected={'M1':22289,'M2':22142,'M3':22035,'M4':22058}[model]
    if len(rows)!=expected:raise RuntimeError(f'{model} training count {len(rows)} != {expected}')
    return IndexedImages(rows,train_tf() if augment else eval_tf(),label)
def restricted_dataset(model):
    order=class_order(model);label={c:i for i,c in enumerate(order)};rows=[r for r in sorted_rows(EVALCSV) if int(r['category_id']) in label]
    if len(rows)!=800:raise RuntimeError('restricted validation count mismatch')
    return IndexedImages(rows,eval_tf(),label)
def reference_dataset():
    ids=set(d_classes());rows=[r for r in sorted_rows(MINICSV) if int(r['category_id']) in ids]
    if len(rows)!=1000:raise RuntimeError('ID reference count mismatch')
    return IndexedImages(rows,eval_tf())
def evaluation_dataset():
    rows=sorted_rows(EVALCSV)
    if len(rows)!=1000:raise RuntimeError('evaluation count mismatch')
    return IndexedImages(rows,eval_tf())

def lr_factor(e):return (e+1)/5 if e<5 else .5*(1+math.cos(math.pi*(e-5)/(300-5)))
def accuracy(model,ds):
    model.eval();right=n=0
    with torch.inference_mode():
        for x,y,_,_ in DataLoader(ds,batch_size=512,num_workers=8,pin_memory=True,persistent_workers=True):y=y.cuda(non_blocking=True);right+=(model(x.cuda(non_blocking=True)).argmax(1)==y).sum().item();n+=len(y)
    return right/n

def preflight():
    ensure();audit=json.loads((ROOT/'canonical_input_audit.json').read_text())
    if audit['status']!='PASS':raise RuntimeError('Canonical input audit must pass')
    init=[]
    for s in SEEDS:
        hs=[state_digest(init_model(s)[0].state_dict()) for _ in MODELS];init.append(hs)
        if len(set(hs))!=1:raise RuntimeError('same-seed initializations differ')
    ds=training_dataset('M1');x,y,_,_=next(iter(DataLoader(ds,batch_size=8,num_workers=0)));mx,soft=mixer()(x,y);m,_=init_model(0);m=m.cuda();m.train();before_step=state_digest(m.state_dict());opt=AdamW(m.parameters(),lr=3e-4,weight_decay=.05);z=m(mx.cuda());loss=F.cross_entropy(z,soft.cuda(),label_smoothing=.1);loss.backward();opt.step();after_step=state_digest(m.state_dict());feature=m.features(mx[:2].cuda());vary=not torch.equal(m.features(mx[:2].cuda()),m.features(mx[:2].cuda()));m.eval();stable=torch.equal(m.features(mx[:2].cuda()),m.features(mx[:2].cuda()))
    p=ROOT/'smoke/roundtrip.pt';torch.save({'state':m.cpu().state_dict(),'pretrained_weights_loaded':False},p);reload,_=init_model(0);reload.load_state_dict(torch.load(p,weights_only=False)['state'],strict=True)
    head=nn.Linear(192,20);net=EncoderProbe(reload,head);sx=torch.randn(2,3,224,224,requires_grad=True);sl=net(sx);grad=torch.autograd.grad(F.cross_entropy(sl/1000,sl.detach().argmax(1)),sx)[0]
    rg=np.random.default_rng(0);synthetic=rg.normal(size=(1000,192));synthetic_labels=np.repeat(np.arange(20),50);query=rg.normal(size=(32,192));w=head.weight.detach().numpy();b=head.bias.detach().numpy();detector_checks=[]
    st,_=fit_mahalanobis(synthetic,synthetic_labels);detector_checks.append(np.isfinite(score_mahalanobis(query,st)).all());st,_=fit_vim(synthetic,w,b);detector_checks.append(np.isfinite(score_vim(query,w,b,st)).all());st,_=fit_neco(synthetic);detector_checks.append(np.isfinite(score_neco(query,st)).all());st,_=fit_nci(synthetic);detector_checks.append(np.isfinite(score_nci(query,w,b,st)).all());st,_=fit_gradorth(synthetic);detector_checks.append(np.isfinite(score_gradorth(query,w,b,st)).all())
    checks={'audit':True,'train_images_nonzero':len(ds)>0,'output_80':z.shape==(8,80),'feature_192':feature.shape==(2,192),'soft_labels':soft.shape==(8,80) and torch.allclose(soft.sum(1),torch.ones(8)),'finite_loss':math.isfinite(float(loss)),'smoke_optimizer_step_changed_weights':before_step!=after_step,'drop_path_training_only':vary and stable,'checkpoint_reload':state_digest(m.state_dict())==state_digest(reload.state_dict()),'gradorth_head_shape':head.weight.shape==(20,192),'odin_gradient_finite':bool(torch.isfinite(grad).all()),'feature_detector_compatibility':bool(all(detector_checks)),'ood_score_orientation':bool(roc_auc_score([0,0,1,1],[0.1,0.2,0.8,0.9])==1.0),'no_pretrained_weights':True,'class_orders_80':all(len(class_order(q))==80 for q in MODELS)}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','created_utc':now(),'checks':checks,'architecture':{'resolution':224,'patch':16,'patches':196,'tokens':197,'depth':12,'hidden':192,'heads':3,'mlp':768,'feature':192,'head_shape':[80,192],'parameters':sum(q.numel() for q in reload.parameters())},'initial_state_hashes':init,'training_images_M1':len(ds),'sampling_policy':'canonical natural imbalance; shuffled without weights or resampling','pretrained_weights_loaded':False,'ood_evaluated':False,'config_sha256':file_hash(CONFIG)};write_json(ROOT/'preflight.json',report)
    if report['status']!='PASS':raise RuntimeError(report)
    print(json.dumps(report,indent=2))

def save_checkpoint(path,payload):q=path.with_suffix('.partial');torch.save(payload,q);os.replace(q,path)
def train(model_id,seed):
    if json.loads((ROOT/'preflight.json').read_text())['status']!='PASS':raise RuntimeError('preflight missing')
    c=cfg();order=class_order(model_id);tag=f'{model_id}_seed{seed}';path=CHECKPOINTS/f'{tag}.pt';m,g=init_model(seed);initial=state_digest(m.state_dict());m=m.cuda();opt=AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);sched=LambdaLR(opt,lr_factor);scaler=torch.amp.GradScaler('cuda',enabled=True);start=0;hist=[]
    if path.exists():
        s=torch.load(path,map_location='cpu',weights_only=False)
        if s['identity']!=[model_id,seed] or s['class_order']!=order or s['config_sha256']!=file_hash(CONFIG):raise RuntimeError('resume mismatch')
        m.load_state_dict(s['model_state']);opt.load_state_dict(s['optimizer_state']);sched.load_state_dict(s['scheduler_state']);scaler.load_state_dict(s['scaler_state']);start=s['epoch'];hist=s['history'];g.set_state(s['generator_state'])
    ds=training_dataset(model_id);loader=DataLoader(ds,batch_size=256,shuffle=True,generator=g,num_workers=12,pin_memory=True,persistent_workers=True,drop_last=False);mix=mixer();began=time.time()
    for e in range(start,300):
        m.train();total=correct=0;ls=0.;t=time.time()
        for x,y,_,_ in loader:
            x,y=mix(x,y);x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=True):z=m(x);loss=F.cross_entropy(z,y,label_smoothing=.1)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=len(y);ls+=float(loss)*len(y);correct+=(z.argmax(1)==y.argmax(1)).sum().item()
        sched.step();row={'epoch':e+1,'soft_target_loss':ls/total,'mixed_target_argmax_accuracy':correct/total,'learning_rate':opt.param_groups[0]['lr'],'epoch_seconds':time.time()-t};hist.append(row);pd.DataFrame(hist).to_csv(TRAINING/f'{tag}.csv',index=False)
        if (e+1)%5==0 or e+1==300:save_checkpoint(path,{'identity':[model_id,seed],'epoch':e+1,'target_epochs':300,'class_order':order,'model_state':{k:v.cpu() for k,v in m.state_dict().items()},'optimizer_state':opt.state_dict(),'scheduler_state':sched.state_dict(),'scaler_state':scaler.state_dict(),'generator_state':g.get_state(),'history':hist,'initial_state_sha256':initial,'config_sha256':file_hash(CONFIG),'pretrained_weights_loaded':False})
        print(f'{tag} epoch={e+1}/300 loss={row["soft_target_loss"]:.4f} mixed_acc={row["mixed_target_argmax_accuracy"]:.4f} lr={row["learning_rate"]:.8f} sec={row["epoch_seconds"]:.1f}',flush=True)
    clean=accuracy(m,training_dataset(model_id,False));test=accuracy(m,restricted_dataset(model_id));meta={'status':'COMPLETE','rotation':model_id,'seed':seed,'train_samples':len(ds),'clean_train_accuracy':clean,'restricted_test_accuracy':test,'train_test_gap':clean-test,'runtime_seconds_current_invocation':time.time()-began,'checkpoint':str(path),'checkpoint_sha256':file_hash(path),'pretrained_weights_loaded':False};write_json(CHECKPOINTS/f'{tag}.json',meta);print(json.dumps(meta))

def classification_audit():
    rows=[]
    for s in SEEDS:
        for m in MODELS:
            x=json.loads((CHECKPOINTS/f'{m}_seed{s}.json').read_text());rows.append({k:x[k] for k in ('rotation','seed','clean_train_accuracy','restricted_test_accuracy','train_test_gap')})
    d=pd.DataFrame(rows);mean=float(d.restricted_test_accuracy.mean());sd=float(d.restricted_test_accuracy.std(ddof=1));span=float(d.restricted_test_accuracy.max()-d.restricted_test_accuracy.min());c=cfg();stable=sd<=c['classification_gate']['stability_max_sd'] and span<=c['classification_gate']['stability_max_range']
    status='INSUFFICIENT' if not stable else ('ADEQUATE' if mean>=c['classification_gate']['adequate_minimum'] else ('MIXED' if mean>=c['classification_gate']['mixed_minimum'] else 'INSUFFICIENT'))
    d.to_csv(CLASSIFICATION/'per_run.csv',index=False);out={'status':status,'mean_restricted_test_accuracy':mean,'sd_restricted_test_accuracy':sd,'min_restricted_test_accuracy':float(d.restricted_test_accuracy.min()),'max_restricted_test_accuracy':float(d.restricted_test_accuracy.max()),'range_restricted_test_accuracy':span,'mean_clean_train_accuracy':float(d.clean_train_accuracy.mean()),'mean_train_test_gap':float(d.train_test_gap.mean()),'canonical_resnet50_mean_restricted_test_accuracy':c['classification_reference']['mean_restricted_test_accuracy'],'difference_from_resnet50_percentage_points':100*(mean-c['classification_reference']['mean_restricted_test_accuracy']),'stability_pass':stable,'gate_thresholds':c['classification_gate'],'ood_evaluated_before_gate':False,'gate_created_utc':now()};write_json(CLASSIFICATION/'classification_gate.json',out);write_gate_report(d,out);print(json.dumps(out,indent=2))

def write_gate_report(runs,cl):
    rr='\n'.join(f'| {x.rotation} | {x.seed:.0f} | {x.clean_train_accuracy:.2%} | {x.restricted_test_accuracy:.2%} | {x.train_test_gap:.2%} |' for x in runs.itertuples())
    stop='OOD evaluation is forbidden and was not run because the gate is INSUFFICIENT.' if cl['status']=='INSUFFICIENT' else 'The classification gate permits the fixed OOD pipeline; no OOD result was inspected before this decision.'
    (ROOT/'REPORT.md').write_text(f"""# iNaturalist 2021 FULL-native ViT architecture robustness experiment

## A. Dataset/design identity
The exact canonical 100-species subset, 20 genus groups, d/c1/c2/c3/c4 roles, M1-M4 rotations, seeds 0/1, native training identities, 1,000 fixed downstream-ID reference identities, and 1,000 fixed validation identities are frozen in `manifests/`. Native imbalance is preserved without weighting or resampling.

## B. Architecture
224x224 input, 16x16 patches, 197 tokens, depth 12, hidden 192, 3 heads, MLP 768, final-LayerNorm 192D CLS feature, 80x192+bias classifier, 5,539,856 parameters. **NO external pretrained weights were used.**

## C. Training recipe
AdamW, lr 3e-4, weight decay .05, 300 epochs, batch 256, five-epoch warmup then cosine decay; label smoothing .1, Mixup .8, CutMix 1.0, RandAugment 2/9, random resized crop, horizontal flip, random erasing .25, drop path .1, AMP, and ImageNet normalization.

## D. Classification-quality audit
| Rotation | Seed | Clean train | Restricted test | Gap |
|---|---:|---:|---:|---:|
{rr}

Mean restricted accuracy {cl['mean_restricted_test_accuracy']:.2%} (SD {cl['sd_restricted_test_accuracy']:.2%}, range {cl['min_restricted_test_accuracy']:.2%}-{cl['max_restricted_test_accuracy']:.2%}); canonical ResNet-50 {cl['canonical_resnet50_mean_restricted_test_accuracy']:.2%}; difference {cl['difference_from_resnet50_percentage_points']:.2f} points. Status **{cl['status']}**. `ood_evaluated_before_gate = false`.

{stop}
""")

def gate():
    p=CLASSIFICATION/'classification_gate.json'
    if not p.exists():raise RuntimeError('classification gate has not run')
    s=json.loads(p.read_text())['status']
    if s=='INSUFFICIENT':raise RuntimeError('classification gate forbids OOD evaluation')
    return s
def load_model(model_id,seed):
    s=torch.load(CHECKPOINTS/f'{model_id}_seed{seed}.pt',map_location='cpu',weights_only=False);m,_=init_model(seed);m.load_state_dict(s['model_state'],strict=True);return m,s
def collect(model,ds):
    fs=[];ls=[];pos=[];ids=[];model=model.cuda().eval();model.requires_grad_(False)
    with torch.inference_mode():
        for x,_,image_id,p in DataLoader(ds,batch_size=512,num_workers=8,pin_memory=True,persistent_workers=True):
            f=model.features(x.cuda(non_blocking=True));fs.append(f.cpu());ls.append(model.classifier(f).cpu());pos.extend(p.tolist());ids.extend(image_id.tolist())
    return torch.cat(fs),torch.cat(ls),pos,ids
def extract(model_id,seed):
    gate();tag=f'{model_id}_seed{seed}';m,s=load_model(model_id,seed);before=state_digest(m.state_dict());refds=reference_dataset();evds=evaluation_dataset();rf,rl,rp,ri=collect(m,refds);ef,el,ep,ei=collect(m,evds);after=state_digest(m.cpu().state_dict());rr=refds.rows;er=evds.rows
    if rp!=list(range(1000)) or ep!=list(range(1000)) or ri!=[int(x['image_id']) for x in rr] or ei!=[int(x['image_id']) for x in er] or before!=after or rf.shape!=(1000,192) or ef.shape!=(1000,192):raise RuntimeError('feature identity/shape invariant failed')
    out=FEATURES/f'{tag}.pt';torch.save({'identity':[model_id,seed],'reference_features':rf,'reference_labels':np.asarray([int(x['category_id']) for x in rr]),'reference_ids':np.asarray(ri),'reference_paths':np.asarray([x['file_name'] for x in rr]),'eval_features':ef,'eval_labels':np.asarray([int(x['category_id']) for x in er]),'eval_ids':np.asarray(ei),'eval_paths':np.asarray([x['file_name'] for x in er]),'reference_upstream_logits':rl,'eval_upstream_logits':el,'feature_dim':192,'state_sha256_before':before,'state_sha256_after':after,'checkpoint_sha256':file_hash(CHECKPOINTS/f'{tag}.pt')},out);np.savez_compressed(FEATURES/f'{tag}_upstream_head.npz',weight=m.classifier.weight.detach().numpy(),bias=m.classifier.bias.detach().numpy(),classes=np.asarray(s['class_order']));write_json(FEATURES/f'{tag}.json',{'status':'PASS','sha256':file_hash(out),'reference_shape':list(rf.shape),'evaluation_shape':list(ef.shape),'fixed_reference_identity':True,'fixed_evaluation_identity':True});print({'status':'PASS','tag':tag})
def train_probe(model_id,seed,data):
    tag=f'{model_id}_seed{seed}';path=PROBES/f'{tag}.pt';dc=sorted(d_classes());mapping={w:i for i,w in enumerate(dc)};x=data['reference_features'].float();y=torch.tensor([mapping[int(w)] for w in data['reference_labels']]);seed_all(seed);head=nn.Linear(192,20);initial=state_digest(head.state_dict());opt=SGD(head.parameters(),lr=.1,momentum=.9,weight_decay=0.);sch=CosineAnnealingLR(opt,50);hist=[];gen=torch.Generator().manual_seed(seed);loader=DataLoader(TensorDataset(x,y),batch_size=512,shuffle=True,generator=gen,num_workers=0)
    for e in range(50):
        total=correct=0;ls=0.
        for f,t in loader:
            opt.zero_grad();z=head(f);loss=F.cross_entropy(z,t);loss.backward();opt.step();total+=len(t);ls+=float(loss)*len(t);correct+=(z.argmax(1)==t).sum().item()
        sch.step();hist.append({'epoch':e+1,'loss':ls/total,'accuracy':correct/total,'learning_rate':opt.param_groups[0]['lr']})
    torch.save({'identity':[model_id,seed],'head':head.state_dict(),'d_classes':dc,'history':hist,'initial_state_sha256':initial,'feature_sha256':file_hash(FEATURES/f'{tag}.pt')},path);mask=np.isin(data['eval_labels'],dc);ey=torch.tensor([mapping[int(w)] for w in data['eval_labels'][mask]]);acc=float((head(data['eval_features'][mask]).argmax(1)==ey).float().mean());write_json(PROBES/f'{tag}.json',{'status':'PASS','downstream_id_test_accuracy':acc,'head_shape':[20,192],'probe_seed':seed,'sha256':file_hash(path)});return head
def knn(q,r):
    r=F.normalize(r.float(),dim=1).cuda();out=[]
    with torch.inference_mode():
        for i in range(0,len(q),256):x=F.normalize(q[i:i+256].float(),dim=1).cuda();out.append((1-x@r.T).topk(50,largest=False,dim=1).values.mean(1).cpu())
    return torch.cat(out).numpy()
def fit_nci(ref):
    x=np.asarray(ref,dtype=np.float64);a=float(cfg()['nci_alpha']);arr={'global_mean':x.mean(0),'alpha':np.asarray(a)};meta={'detector':'nci','feature_dim':192,'reference_count':len(x),'alpha':a,'alpha_selection':cfg()['nci_rationale']};meta['fit_state_sha256']=detector_state_hash(arr,meta);return arr,meta
def save_fit(tag,name,arr,meta,note):
    d=FITS/tag;d.mkdir(exist_ok=True);np.savez_compressed(d/f'{name}.npz',**arr);write_json(d/f'{name}.json',{**meta,'status':'PASS','fit_data':'canonical 1,000 downstream-ID mini-train images only','focal_ood_used_for_fit_or_tuning':False,'architecture_adaptation':note})
def odin_batch(net,x):
    x=x.detach().requires_grad_(True);z=net(x);pred=z.detach().argmax(1);g=torch.autograd.grad(F.cross_entropy(z/1000,pred,reduction='sum'),x)[0];pert=x.detach()-.0014*torch.where(g>=0,1.,-1.)/torch.tensor(cfg()['normalization_std'],device='cuda').view(1,3,1,1)
    with torch.inference_mode():score=1-torch.softmax(net(pert)/1000,1).max(1).values
    return score,g,pert
def odin_scores(model_id,seed,head,cached):
    m,_=load_model(model_id,seed);net=EncoderProbe(m,head).cuda().eval();net.requires_grad_(False);before=state_digest(net.state_dict());vals=[];positions=[];audit=None
    for bi,(x,_,_,p) in enumerate(DataLoader(evaluation_dataset(),batch_size=256,num_workers=8,pin_memory=True,persistent_workers=True)):
        x=x.cuda(non_blocking=True)
        if bi==0:
            with torch.inference_mode():raw=net.features(x)
            expected=cached['eval_features'][p].cuda();match=float(F.cosine_similarity(raw,expected).min())
        sc,g,pert=odin_batch(net,x);vals.append(sc.cpu().numpy());positions.extend(p.tolist())
        if bi==0:audit={'minimum_cached_feature_cosine':match,'gradient_finite':bool(torch.isfinite(g).all()),'raw_pixel_step':torch.mean(torch.abs((pert-x)*torch.tensor(cfg()['normalization_std'],device='cuda').view(1,3,1,1)),dim=(0,2,3)).tolist()}
    out=np.concatenate(vals);after=state_digest(net.cpu().state_dict())
    if positions!=list(range(1000)) or before!=after or not np.isfinite(out).all() or audit['minimum_cached_feature_cosine']<.99999:raise RuntimeError('ODIN invariant failed')
    return out,{'temperature':1000.,'epsilon_raw_pixel_units':.0014,'precision':'FP32','clipping':False,'first_batch':audit,'parameters_frozen':True,'focal_ood_used_for_tuning':False}
def evaluate(model_id,seed):
    gate();tag=f'{model_id}_seed{seed}';data=torch.load(FEATURES/f'{tag}.pt',map_location='cpu',weights_only=False);head=train_probe(model_id,seed,data);ref=data['reference_features'].numpy();ev=data['eval_features'].numpy();labels=data['reference_labels'];w=head.weight.detach().numpy();b=head.bias.detach().numpy();logits=probe_logits(ev,w,b);lt=torch.from_numpy(logits);scores={'knn':knn(data['eval_features'],data['reference_features']),'energy':(-torch.logsumexp(lt,1)).numpy(),'msp':(1-lt.softmax(1).max(1).values).numpy()}
    st,me=fit_mahalanobis(ref,labels);scores['mahalanobis']=score_mahalanobis(ev,st);save_fit(tag,'mahalanobis',st,me,'192D final CLS feature; tied covariance fit is ID-only')
    st,me=fit_vim(ref,w,b);scores['vim']=score_vim(ev,w,b,st);save_fit(tag,'vim',st,me,'192D CLS and 20x192 probe; canonical rule gives 96D principal/residual split')
    st,me=fit_neco(ref);me['architecture_branch']='ViT CLS';scores['neco']=score_neco(ev,st);save_fit(tag,'neco',st,me,'unchanged 90% ID explained-variance rule')
    st,me=fit_nci(ref);scores['nci']=score_nci(ev,w,b,st);save_fit(tag,'nci',st,me,'fixed alpha=0.01 chosen for 192D ViT before OOD evaluation; no sweep')
    st,me=fit_gradorth(ref);scores['gradorth']=score_gradorth(ev,w,b,st);save_fit(tag,'gradorth',st,me,'192D CLS and 20x192 probe; unchanged full-reference 97% SVD rule')
    scores['odin'],om=odin_scores(model_id,seed,head,data);write_json(FITS/tag/'odin.json',{**om,'status':'PASS','fit_data':'no fit; fixed canonical transfer values','architecture_adaptation':'input gradient through ViT CLS encoder; canonical iNaturalist T=1000 and epsilon=.0014 unchanged','focal_ood_used_for_fit_or_tuning':False})
    if set(scores)!=set(DETECTORS) or any(len(x)!=1000 or not np.isfinite(x).all() for x in scores.values()):raise RuntimeError('score coverage failed')
    np.savez_compressed(SCORES/f'{tag}.npz',class_ids=data['eval_labels'],evaluation_ids=data['eval_ids'],downstream_logits=logits,**scores);write_json(SCORES/f'{tag}.json',{'status':'PASS','detectors':list(DETECTORS),'evaluation_images':1000,'reference_images':1000,'sha256':file_hash(SCORES/f'{tag}.npz')});print({'status':'PASS','tag':tag})
def aggregate():
    gate();cm=candidates();dc=np.asarray(d_classes());states=[]
    for m in MODELS:
      for s in SEEDS:
        z=np.load(SCORES/f'{m}_seed{s}.npz');y=z['class_ids'];idm=np.isin(y,dc)
        if idm.sum()!=200:raise RuntimeError('ID evaluation count mismatch')
        for cid,item in cm.items():
          om=y==cid
          if om.sum()!=10:raise RuntimeError('OOD evaluation count mismatch')
          for det in DETECTORS:states.append({'dataset':'iNaturalist 2021 FULL-native','architecture':'ViT-Tiny/16','detector':det,**item,'seed':s,'rotation':m,'state':'withheld' if m==item['withheld_model'] else 'present','id_eval_images':200,'ood_eval_images':10,'auroc':float(roc_auc_score(np.r_[np.zeros(idm.sum()),np.ones(om.sum())],np.r_[z[det][idm],z[det][om]]))})
    sf=pd.DataFrame(states);sf.to_csv(SUMMARIES/'per_state_aurocs.csv',index=False);sr=[]
    keys=['detector','group_id','group_name','class_id','class_name','role','withheld_model','seed']
    for vals,g in sf.groupby(keys):
      x=dict(zip(keys,vals));wh=g[g.state=='withheld'];pr=g[g.state=='present'];by=g.set_index('rotation').auroc.to_dict()
      if len(wh)!=1 or len(pr)!=3:raise RuntimeError('paired aggregation failed')
      sr.append({**x,'auroc_withheld':float(wh.auroc.iloc[0]),'auroc_supervised':float(pr.auroc.mean()),'delta':float(wh.auroc.iloc[0]-pr.auroc.mean()),**{f'auroc_{m}':by[m] for m in MODELS}})
    se=pd.DataFrame(sr);se.to_csv(SUMMARIES/'seed_level_effects.csv',index=False);cr=[];ck=keys[:-1]
    for vals,g in se.groupby(ck):cr.append({**dict(zip(ck,vals)),'delta_seed0':float(g[g.seed==0].delta.iloc[0]),'delta_seed1':float(g[g.seed==1].delta.iloc[0]),'delta':float(g.delta.mean()),'mean_auroc_withheld':float(g.auroc_withheld.mean()),'mean_auroc_supervised':float(g.auroc_supervised.mean())})
    cf=pd.DataFrame(cr);cf.to_csv(SUMMARIES/'class_level_effects.csv',index=False);gg=cf.groupby(['detector','group_id','group_name'],as_index=False).agg(classes=('class_id','size'),mean_delta=('delta','mean'));gg.to_csv(SUMMARIES/'group_level_effects.csv',index=False);summary=[];ss=[]
    for det in DETECTORS:
      x=cf[cf.detector==det];v=x.delta.to_numpy();gv=gg[gg.detector==det].mean_delta.to_numpy();rng=np.random.default_rng(cfg()['bootstrap_seed']);draw=gv[rng.integers(0,20,(10000,20))].mean(1);pd.DataFrame({'draw':np.arange(1,10001),'mean_delta':draw}).to_csv(BOOTSTRAP/f'{det}.csv',index=False)
      summary.append({'detector':det,'detector_display':DISPLAY[det],'mean_delta':float(v.mean()),'ci95_low':float(np.quantile(draw,.025)),'ci95_high':float(np.quantile(draw,.975)),'negative_classes':int((v<0).sum()),'positive_classes':int((v>0).sum()),'negative_groups':int((gv<0).sum()),'positive_groups':int((gv>0).sum()),'min_delta':float(v.min()),'q1_delta':float(np.quantile(v,.25)),'median_delta':float(np.median(v)),'q3_delta':float(np.quantile(v,.75)),'max_delta':float(v.max()),'mean_auroc_withheld':float(x.mean_auroc_withheld.mean()),'mean_auroc_supervised':float(x.mean_auroc_supervised.mean())})
      for s in SEEDS:
        q=se[(se.detector==det)&(se.seed==s)].delta.to_numpy();ss.append({'detector':det,'seed':s,'mean_delta':float(q.mean()),'negative_classes':int((q<0).sum()),'positive_classes':int((q>0).sum())})
    sm=pd.DataFrame(summary);sm.to_csv(SUMMARIES/'detector_summary.csv',index=False);pd.DataFrame(ss).to_csv(SUMMARIES/'seed_specific_summary.csv',index=False);rv=[]
    for cid,item in cm.items():
      q=se[se.class_id==cid];gw=float(q[q.detector=='knn'].auroc_withheld.mean()-q[q.detector=='energy'].auroc_withheld.mean());gp=float(q[q.detector=='knn'].auroc_supervised.mean()-q[q.detector=='energy'].auroc_supervised.mean());rv.append({**item,'gap_withheld':gw,'gap_supervised':gp,'reversal':gw*gp<0,'tie':gw==0 or gp==0})
    rv=pd.DataFrame(rv);rv.to_csv(SUMMARIES/'knn_energy_ranking_reversals.csv',index=False);wide=cf.pivot(index=['group_id','group_name','class_id','class_name'],columns='detector',values='delta').reset_index();wide['sign_disagreement']=[len(set(np.sign(r)))>1 for r in wide[list(DETECTORS)].to_numpy()];wide.to_csv(SUMMARIES/'cross_detector_sign_disagreement.csv',index=False)
    rn=pd.read_csv(EXPANDED/'detector_dataset_summary.csv').query("dataset_slug=='inat'");comp=sm.merge(rn[['detector','mean_delta','ci95_low','ci95_high','negative_classes','negative_groups']],on='detector',suffixes=('_vit','_resnet'));comp.to_csv(SUMMARIES/'resnet50_vs_vit.csv',index=False);result={'status':'PASS','created_utc':now(),'vit_knn_energy_reversals':int(rv.reversal.sum()),'vit_knn_energy_ties':int(rv.tie.sum()),'resnet50_knn_energy_reversals':24,'vit_cross_detector_sign_disagreement':int(wide.sign_disagreement.sum()),'resnet50_cross_detector_sign_disagreement':70,'detectors':summary};write_json(SUMMARIES/'results.json',result);cross=cross_dataset_summary(sm);make_report(sm,pd.DataFrame(ss),comp,result,cross);validate(sf,se,cf,gg,sm);print(json.dumps(result,indent=2))
def cross_dataset_summary(inat_sm):
    expanded=pd.read_csv(EXPANDED/'detector_dataset_summary.csv');rows=[]
    specs=[('CIFAR-100','cifar100','ResNet-18',PROJECT/'cifar100_vit_v2_generalization'),('controlled ImageNet','imagenet','ResNet-50',PROJECT/'imagenet_vit_architecture_robustness'),('iNaturalist 2021 FULL-native','inat','ResNet-50',ROOT)]
    resnet_acc={'cifar100':.7818,'imagenet':None,'inat':cfg()['classification_reference']['mean_restricted_test_accuracy']}
    for dataset,slug,rarch,vroot in specs:
      r=expanded[(expanded.dataset_slug==slug)&(expanded.detector=='knn')].iloc[0];v=inat_sm[inat_sm.detector=='knn'].iloc[0] if slug=='inat' else pd.read_csv(vroot/'summaries/detector_summary.csv').query("detector=='knn'").iloc[0];cl=json.loads((vroot/'classification_results/classification_gate.json').read_text())
      rows.append({'dataset':dataset,'resnet_architecture':rarch,'vit_architecture':'ViT-Tiny/4' if slug=='cifar100' else 'ViT-Tiny/16','resnet_knn_mean_delta':r.mean_delta,'vit_knn_mean_delta':v.mean_delta,'resnet_negative_classes':int(r.negative_classes),'vit_negative_classes':int(v.negative_classes),'resnet_negative_groups':int(r.negative_groups),'vit_negative_groups':int(v.negative_groups),'resnet_classification_accuracy':resnet_acc[slug],'vit_classification_accuracy':cl['mean_restricted_test_accuracy']})
    d=pd.DataFrame(rows);d.to_csv(SUMMARIES/'cross_dataset_architecture_summary.csv',index=False);return d

def make_report(sm,ss,comp,result,cross):
    cl=json.loads((CLASSIFICATION/'classification_gate.json').read_text());runs=pd.read_csv(CLASSIFICATION/'per_run.csv');rows=[]
    for d in DETECTORS:
      x=sm[sm.detector==d].iloc[0];r=comp[comp.detector==d].iloc[0];rows.append(f"| {DISPLAY[d]} | {r.mean_delta_resnet:.6f} | [{r.ci95_low_resnet:.6f}, {r.ci95_high_resnet:.6f}] | {x.mean_delta:.6f} | [{x.ci95_low:.6f}, {x.ci95_high:.6f}] | {x.negative_classes:.0f}/80 | {x.negative_groups:.0f}/20 | {x.mean_auroc_withheld:.6f} | {x.mean_auroc_supervised:.6f} |")
    rr=[f"| {x.rotation} | {x.seed:.0f} | {x.clean_train_accuracy:.2%} | {x.restricted_test_accuracy:.2%} | {x.train_test_gap:.2%} |" for x in runs.itertuples()];k=sm[sm.detector=='knn'].iloc[0];ks=ss[ss.detector=='knn'];cross_rows=[]
    for x in cross.itertuples():cross_rows.append(f"| {x.dataset} | {x.resnet_knn_mean_delta:.6f} | {x.vit_knn_mean_delta:.6f} | {x.resnet_negative_classes}/80 | {x.vit_negative_classes}/80 | {x.resnet_negative_groups}/20 | {x.vit_negative_groups}/20 | {'unavailable' if pd.isna(x.resnet_classification_accuracy) else f'{x.resnet_classification_accuracy:.2%}'} | {x.vit_classification_accuracy:.2%} |")
    text=f'''# iNaturalist 2021 FULL-native ViT architecture robustness experiment

## A. Dataset/design identity
The exact canonical 100-species subset, 20 genus groups, d/c1/c2/c3/c4 assignments, M1-M4 rotations, seeds 0/1, 27,713 native training identities, 1,000 downstream-ID reference identities, and 1,000 validation identities were reused. Native imbalance is preserved without weighting or resampling.

## B. ViT architecture
224x224 input, 16x16 patches, 196 image patches and 197 tokens, depth 12, hidden 192, 3 heads, MLP 768, final-LayerNorm 192D CLS feature, 80x192+bias classifier, 5,539,856 parameters. **NO external pretrained weights were used.**

## C. Training recipe
AdamW, lr 3e-4, weight decay .05, 300 epochs, batch/effective batch 256, five-epoch warmup and cosine decay; label smoothing .1, Mixup .8, CutMix 1.0, RandAugment 2/9, random resized crop, horizontal flip, random erasing .25, drop path .1, and ImageNet normalization.

## D. Classification-quality audit
| Rotation | Seed | Clean train | Restricted test | Gap |
|---|---:|---:|---:|---:|
{chr(10).join(rr)}

Mean restricted accuracy {cl['mean_restricted_test_accuracy']:.2%} (SD {cl['sd_restricted_test_accuracy']:.2%}, range {cl['min_restricted_test_accuracy']:.2%}-{cl['max_restricted_test_accuracy']:.2%}); canonical ResNet-50 {cl['canonical_resnet50_mean_restricted_test_accuracy']:.2%}; difference {cl['difference_from_resnet50_percentage_points']:.2f} points. Status **{cl['status']}**. `ood_evaluated_before_gate = false`.

## E. Primary kNN result
Mean Delta {k.mean_delta:.6f}, 20-genus bootstrap 95% CI [{k.ci95_low:.6f}, {k.ci95_high:.6f}], {k.negative_classes:.0f}/80 negative species and {k.negative_groups:.0f}/20 negative genus means. Seed 0: {ks[ks.seed==0].mean_delta.iloc[0]:.6f}, {ks[ks.seed==0].negative_classes.iloc[0]:.0f}/80 negative. Seed 1: {ks[ks.seed==1].mean_delta.iloc[0]:.6f}, {ks[ks.seed==1].negative_classes.iloc[0]:.0f}/80 negative. Withheld AUROC {k.mean_auroc_withheld:.6f}; supervised AUROC {k.mean_auroc_supervised:.6f}.

## F. Full detector table
| Detector | ResNet-50 Delta | ResNet 95% CI | ViT Delta | ViT 95% CI | Negative species | Negative groups | Withheld AUROC | Supervised AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## G. Detector heterogeneity
ResNet-50 versus ViT kNN-Energy reversals: 24/80 versus {result['vit_knn_energy_reversals']}/80; ViT ties {result['vit_knn_energy_ties']}. Cross-detector sign disagreement: 70/80 versus {result['vit_cross_detector_sign_disagreement']}/80.

## H. Cross-dataset architecture summary
| Dataset | ResNet kNN Delta | ViT kNN Delta | ResNet negative classes | ViT negative classes | ResNet negative groups | ViT negative groups | ResNet classification | ViT classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cross_rows)}

## I. Compatibility notes
Mahalanobis uses the 192D CLS feature and ID-only tied covariance. ViM reads the 20x192 probe mechanically and uses an ID-only 96D principal/residual split and alpha. NECO retains the ID-only 90% rule. NCI uses fixed alpha .01 established for 192D ViT before iNaturalist OOD evaluation. GradOrth retains the deterministic full-reference 97% SVD rule. ODIN retains T=1000 and epsilon=.0014 in FP32. All fitted quantities use only the canonical 1,000 downstream-ID mini-train images; focal OOD data were not used for fitting or tuning.

## J. Scientific interpretation
The classification audit establishes control quality. The kNN result addresses primary replication, the detector table addresses broader provenance sensitivity, and the reversal/disagreement counts describe detector-specific architecture dependence. Architecture robustness does not require equal effect sizes or detector signs and does not establish architecture invariance or mechanism.
''';(ROOT/'REPORT.md').write_text(text)
def validate(sf,se,cf,gg,sm):
    checks={'checkpoints':len(list(CHECKPOINTS.glob('M*_seed*.pt')))==8,'checkpoint_metadata':len(list(CHECKPOINTS.glob('M*_seed*.json')))==8,'features':len(list(FEATURES.glob('M*_seed*.pt')))==8,'scores':len(list(SCORES.glob('M*_seed*.npz')))==8,'states':len(sf)==5760,'seed_effects':len(se)==1440,'class_effects':len(cf)==720,'group_effects':len(gg)==180,'detectors':len(sm)==9,'fits':len(list(FITS.glob('M*_seed*/*.json')))==48,'no_pretrained':all(json.loads(p.read_text())['pretrained_weights_loaded'] is False for p in CHECKPOINTS.glob('M*_seed*.json')),'canonical_audit':json.loads((ROOT/'canonical_input_audit.json').read_text())['status']=='PASS','ood_after_gate':(CLASSIFICATION/'classification_gate.json').stat().st_mtime<=min(p.stat().st_mtime for p in SCORES.glob('M*_seed*.npz'))};write_json(ROOT/'validation.json',{'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'created_utc':now(),'pretrained_weights_loaded':False,'focal_ood_used_for_detector_tuning':False})

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True);sp.add_parser('preflight');sp.add_parser('classification-audit');sp.add_parser('aggregate')
    for q in ('train','extract','evaluate'):a=sp.add_parser(q);a.add_argument('--model',choices=MODELS,required=True);a.add_argument('--seed',choices=SEEDS,type=int,required=True)
    a=p.parse_args();record()
    if a.cmd=='preflight':preflight()
    elif a.cmd=='train':train(a.model,a.seed)
    elif a.cmd=='extract':extract(a.model,a.seed)
    elif a.cmd=='evaluate':evaluate(a.model,a.seed)
    elif a.cmd=='aggregate':aggregate()
    else:classification_audit()
if __name__=='__main__':main()
