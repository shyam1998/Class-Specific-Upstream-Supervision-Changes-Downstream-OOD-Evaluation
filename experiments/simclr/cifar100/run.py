#!/usr/bin/env python3
"""CIFAR-100 grouped-rotation SimCLR training, extraction, and linear probing."""
from __future__ import annotations
import argparse, copy, csv, json, math, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import CIFAR100
from torchvision import transforms as T
from torchvision.models import resnet18

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
DATA=Path(os.environ.get("CIFAR100_ROOT",REPO/"data/cifar100"))
MANIFEST=REPO/"manifests/cifar100/manifest.json"; CONFIG=ROOT/"config.json"
MODELS=("M1","M2","M3","M4"); SEEDS=(0,1)
MEAN=(.5071,.4867,.4408); STD=(.2675,.2565,.2761)

def cfg():return json.loads(CONFIG.read_text())
def design():return json.loads(MANIFEST.read_text())
def set_seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def atomic_save(x,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');torch.save(x,q);os.replace(q,p)
def classes(model):
 out=[]
 for g in design()["groups"]:
  out.append(int(g["d"]["fine_id"]))
  out.extend(int(g[r]["fine_id"]) for r in ("c1","c2","c3","c4") if g[r]["withheld_model"]!=model)
 return out

def backbone():
 m=resnet18(weights=None);m.conv1=nn.Conv2d(3,64,3,1,1,bias=False);m.maxpool=nn.Identity();m.fc=nn.Identity();return m
class SimCLR(nn.Module):
 def __init__(self):super().__init__();self.encoder=backbone();self.projector=nn.Sequential(nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,128))
 def forward(self,x):
  h=self.encoder(x);return h,F.normalize(self.projector(h),dim=1)
class Remap(Dataset):
 def __init__(self,train,keep,transform):
  self.base=CIFAR100(DATA,train=train,download=False);self.indices=[i for i,y in enumerate(self.base.targets) if y in keep];self.map={y:i for i,y in enumerate(keep)};self.transform=transform
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  j=self.indices[i];im=self.base.data[j];y=self.base.targets[j];return self.transform(im),self.map[y],j,y
class TwoView:
 def __init__(self,t):self.t=t
 def __call__(self,x):return self.t(x),self.t(x)
def aug():
 a=cfg()["augmentation"];return T.Compose([T.ToPILImage(),T.RandomResizedCrop(32,scale=tuple(a["random_resized_crop"]["scale"])),T.RandomHorizontalFlip(a["horizontal_flip_probability"]),T.RandomApply([T.ColorJitter(*a["color_jitter"])],p=a["color_jitter_probability"]),T.RandomGrayscale(a["grayscale_probability"]),T.ToTensor(),T.Normalize(MEAN,STD)])
def clean():return T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize(MEAN,STD)])
def nt_xent(a,b,temp):
 z=torch.cat((a,b));logits=z@z.T/temp;logits.fill_diagonal_(float('-inf'));target=(torch.arange(len(z),device=z.device)+len(a))%len(z);return F.cross_entropy(logits,target)
def train(model,seed):
 c=cfg();set_seed(seed);net=SimCLR().cuda();initial=copy.deepcopy(net.state_dict());ds=Remap(True,classes(model),TwoView(aug()));gen=torch.Generator().manual_seed(seed);dl=DataLoader(ds,batch_size=c["batch_size"],shuffle=True,drop_last=True,num_workers=8,pin_memory=True,persistent_workers=True,generator=gen);opt=torch.optim.SGD(net.parameters(),lr=c["learning_rate"],momentum=c["momentum"],weight_decay=c["weight_decay"]);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c["epochs"])
 log=[]
 for epoch in range(c["epochs"]):
  net.train();losses=[]
  for (x1,x2),_,_,_ in dl:
   x1=x1.cuda(non_blocking=True);x2=x2.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);_,z1=net(x1);_,z2=net(x2);loss=nt_xent(z1,z2,c["temperature"]);loss.backward();opt.step();losses.append(float(loss))
  sch.step();log.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"lr":opt.param_groups[0]["lr"]})
 out=ROOT/"checkpoints"/f"{model}_seed{seed}.pt";atomic_save({"encoder":net.encoder.state_dict(),"projector":net.projector.state_dict(),"initial_state":initial,"model":model,"seed":seed,"epoch":c["epochs"],"external_pretrained_weights":False},out);(ROOT/"training_metrics").mkdir(exist_ok=True);(ROOT/"training_metrics"/f"{model}_seed{seed}.json").write_text(json.dumps(log,indent=2)+'\n')
@torch.inference_mode()
def encode(net,ds):
 dl=DataLoader(ds,batch_size=512,shuffle=False,num_workers=8,pin_memory=True);fs=[];ys=[];ids=[]
 net.eval().cuda()
 for x,_,i,y in dl:fs.append(net(x.cuda(non_blocking=True)).cpu());ys.append(y);ids.append(i)
 return torch.cat(fs),torch.cat(ys),torch.cat(ids)
def extract(model,seed):
 p=torch.load(ROOT/"checkpoints"/f"{model}_seed{seed}.pt",map_location='cpu',weights_only=False);net=backbone();net.load_state_dict(p["encoder"]);d={int(g["d"]["fine_id"]) for g in design()["groups"]};train_ds=Remap(True,sorted(d),clean());test_ds=Remap(False,list(range(100)),clean());tf,ty,ti=encode(net,train_ds);ef,ey,ei=encode(net,test_ds);atomic_save({"train_id_features":tf,"train_id_labels":ty,"train_id_indices":ti,"test_features":ef,"test_labels":ey,"test_indices":ei,"model":model,"seed":seed},ROOT/"features"/f"{model}_seed{seed}.pt")
def probe(model,seed):
 c=cfg()["probe"];f=torch.load(ROOT/"features"/f"{model}_seed{seed}.pt",map_location='cpu',weights_only=False);x=f["train_id_features"];y=f["train_id_labels"];order=sorted(set(y.tolist()));mp={v:i for i,v in enumerate(order)};y=torch.tensor([mp[int(v)] for v in y]);set_seed(10000+seed);head=nn.Linear(512,20).cuda();opt=torch.optim.SGD(head.parameters(),lr=c["learning_rate"],momentum=c["momentum"],weight_decay=c["weight_decay"]);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c["epochs"]);dl=DataLoader(TensorDataset(x,y),batch_size=c["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(10000+seed))
 for _ in range(c["epochs"]):
  for a,b in dl:opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(a.cuda()),b.cuda());loss.backward();opt.step()
  sch.step()
 atomic_save({"head":head.cpu().state_dict(),"class_order":order,"model":model,"seed":seed},ROOT/"probes"/f"{model}_seed{seed}.pt")
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("train","extract","probe","all"));p.add_argument('--model',choices=MODELS);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args();jobs=[(a.model,a.seed)] if a.model is not None and a.seed is not None else [(m,s) for s in SEEDS for m in MODELS]
 for m,s in jobs:
  if a.stage in ('train','all'):train(m,s)
  if a.stage in ('extract','all'):extract(m,s)
  if a.stage in ('probe','all'):probe(m,s)
if __name__=='__main__':main()
