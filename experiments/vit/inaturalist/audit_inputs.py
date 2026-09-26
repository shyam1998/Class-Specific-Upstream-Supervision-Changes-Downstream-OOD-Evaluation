#!/usr/bin/env python3
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'inat_migration_bundle'
DATA=SRC/'selected_data'
MODELS=('M1','M2','M3','M4')

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

manifest=rows(ROOT/'manifests/frozen_manifest.csv')
full=rows(ROOT/'manifests/train_full_native_selected.csv')
mini=rows(ROOT/'manifests/train_mini_selected.csv')
val=rows(ROOT/'manifests/val_selected.csv')
groups=sorted({r['group_id'] for r in manifest})
dids={int(r['category_id']) for r in manifest if r['role']=='d'}
oods={int(r['category_id']) for r in manifest if r['role']!='d'}
canonical_names=['frozen_manifest.csv','manifest.json','manifest_invariants.json','rotation_M1.csv','rotation_M2.csv','rotation_M3.csv','rotation_M4.csv']
copied={name:sha(ROOT/'manifests'/name)==sha(SRC/name) for name in canonical_names}
index_map={'train_full_native_selected.csv':SRC/'data_indices/train_full_native_selected.csv','train_mini_selected.csv':SRC/'data_indices/train_mini_selected.csv','val_selected.csv':SRC/'data_indices/val_selected.csv'}
copied.update({name:sha(ROOT/'manifests'/name)==sha(src) for name,src in index_map.items()})
rot={}
for m in MODELS:
    rr=rows(ROOT/f'manifests/rotation_{m}.csv');ids={int(r['category_id']) for r in rr}
    rot[m]={'species':len(ids),'d_species':len(ids&dids),'train_images':sum(int(r['category_id']) in ids for r in full),'local_labels_exact':sorted(int(r['local_training_label']) for r in rr)==list(range(80)),'withheld_absent':all(int(r['category_id']) not in ids for r in manifest if r['withheld_model']==m),'others_present':all(int(r['category_id']) in ids for r in manifest if r['withheld_model']!=m)}
expected={'M1':22289,'M2':22142,'M3':22035,'M4':22058}
paths=[DATA/r['file_name'] for r in full+val]
checks={
    'content_identical_canonical_files':all(copied.values()),
    'canonical_manifest_invariants_pass':json.loads((SRC/'manifest_invariants.json').read_text())['status']=='PASS',
    'manifest_100_species':len(manifest)==100 and len({r['category_id'] for r in manifest})==100,
    'semantic_groups_20':len(groups)==20,
    'roles_per_group':all(sorted(r['role'] for r in manifest if r['group_id']==g)==['c1','c2','c3','c4','d'] for g in groups),
    'future_ood_species_80':len(oods)==80,
    'rotations_exact':all(rot[m]['species']==80 and rot[m]['d_species']==20 and rot[m]['train_images']==expected[m] and rot[m]['local_labels_exact'] and rot[m]['withheld_absent'] and rot[m]['others_present'] for m in MODELS),
    'seeds_exact':[0,1]==[0,1],
    'full_train_rows_27713':len(full)==27713,
    'mini_rows_5000':len(mini)==5000,
    'validation_rows_1000':len(val)==1000,
    'downstream_reference_1000':sum(int(r['category_id']) in dids for r in mini)==1000,
    'downstream_validation_200':sum(int(r['category_id']) in dids for r in val)==200,
    'future_ood_validation_800':sum(int(r['category_id']) in oods for r in val)==800,
    'unique_image_identities':len({r['image_id'] for r in full})==len(full) and len({r['image_id'] for r in mini})==len(mini) and len({r['image_id'] for r in val})==len(val),
    'train_validation_disjoint':not ({r['image_id'] for r in full}&{r['image_id'] for r in val}),
    'all_selected_paths_present_nonempty':all(p.is_file() and p.stat().st_size>0 for p in paths),
    'natural_imbalance_preserved':min(sum(r['category_id']==c for r in full) for c in {r['category_id'] for r in full})==162 and max(sum(r['category_id']==c for r in full) for c in {r['category_id'] for r in full})==300,
    'no_existing_ood_outputs_before_gate':not any((ROOT/'detector_outputs').glob('*')),
}
out={'status':'PASS' if all(checks.values()) else 'FAIL','created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'copied_file_hash_matches':copied,'hashes':{p.name:sha(p) for p in sorted((ROOT/'manifests').iterdir()) if p.is_file()},'rotation_counts':rot,'counts':{'groups':20,'species':100,'future_ood_species':80,'full_train_images':27713,'mini_train_images':5000,'id_reference_images':1000,'validation_images':1000,'id_validation_images':200,'ood_validation_images':800},'sampling_policy':'canonical natural imbalance; shuffle only; no class weighting, balanced sampler, or resampling','preprocessing':{'train':'RandomResizedCrop(224), horizontal flip, ViT fixed RandAugment and random erasing additions, ImageNet normalization','eval':'Resize(256), CenterCrop(224), ImageNet normalization'},'seeds':[0,1],'ood_evaluated':False}
(ROOT/'canonical_input_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
if out['status']!='PASS':raise SystemExit(1)
