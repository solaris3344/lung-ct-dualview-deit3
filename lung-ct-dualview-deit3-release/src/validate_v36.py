#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3.6 frozen-model refit and locked holdout/external validation.

DEV = 558 train+valid images; holdout = disjoint 288-image test, never used
for fitting/selecting. Clinical external work requires curated, patient-labelled
PNG/JPG inputs and is intentionally reported separately from four-class testing.

This is a final-refit of V3.4 architecture, NOT the exact 15-fold V3.4 ensemble.
No claimed performance result is manufactured by this script.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score)

from v34_reference import (SIGNATURE, VIEW_NAMES, LABELS, CLASS_SHORT, MODEL_ID,
    read_csv, save_json, load_data, crop, sha_text)

SEEDS = (42,52,62)
VERSION = 'v36-final-refit-2026-10-25'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def normalized_label(folder):
    key=folder.lower().replace('\\','/').split('/')[-1].split('_')[0]
    if key not in LABELS:raise ValueError(f'Unknown class folder {folder!r}; expected first part one of {LABELS}')
    return LABELS.index(key)


def inventory(a):
    # In original V3.4, --keep-final-weights default False and resume checkpoint
    # removed after fold predictions. Never pretend OOF arrays can infer new images.
    p=a.root/'runs/adaptive_region_v34_20ep_3seed'
    w=list(p.glob('weights_seed*_fold*.pt')) if p.exists() else []
    ck=list(p.glob('resume_seed*_fold*.pt')) if p.exists() else []
    pred=list(p.glob('pred_seed*_fold*.npz')) if p.exists() else []
    report=p/'report_v34.json'
    result={'v34_out':str(p),'v34_report_exists':report.is_file(),
            'saved_fold_weights_count':len(w), 'saved_resume_count':len(ck),
            'saved_fold_predictions_count':len(pred),
            'final_seed_weights':{str(s):str(a.out/f'final_seed{s}.pt') if (a.out/f'final_seed{s}.pt').is_file() else None for s in SEEDS},
            'important':'V3.4 OOF predictions cannot be used to predict 288 holdout images. Final-refit of 3 models is separate from prior 15-fold ensemble.'}
    if report.exists():
        d=json.loads(report.read_text(encoding='utf-8'))
        m=d.get('methods',{}).get('internal_fixed_15',{})
        result['original_v34_heldout_DEV_only']={'correct':m.get('correct'),'acc':m.get('acc'),'epochs':d.get('epochs'),'seeds':d.get('seeds')}
    return result


def img_read(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def dhash(path):
    from PIL import Image
    im=img_read(path).convert('L').resize((9,8),Image.Resampling.LANCZOS)
    x=np.asarray(im,dtype=np.int16)
    v=0
    for flag in (x[:,1:]>x[:,:-1]).reshape(-1):v=(v<<1)|int(flag)
    return v


def build_test_inventory(a, dev_rel, expected=288):
    root=a.data_root/'test'
    if not root.is_dir():raise FileNotFoundError(f'Missing untouched test directory {root}')
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.webp'):continue
        rel=p.relative_to(a.data_root).as_posix()
        yy=normalized_label(p.parent.name)
        rows.append({'relative_path':rel,'image_path':str(p),'label':yy,'sha256':sha_file(p)})
    if len(rows)!=expected:raise ValueError(f'Expected exactly {expected} test images; found {len(rows)}; abort until source is audited')
    if len({r['relative_path'] for r in rows})!=len(rows):raise ValueError('Repeated test names')
    dev_paths=[a.data_root/s for s in dev_rel]
    missing=[str(p) for p in dev_paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'DEV files missing (e.g. {missing[0]})')
    dev_sha={sha_file(p) for p in dev_paths}
    exact=[r['relative_path'] for r in rows if r['sha256'] in dev_sha]
    # Image similarity is only a WARNING, not sufficient evidence of same patient.
    dev_h=[dhash(p) for p in dev_paths]
    near=[]
    for r in rows:
        q=dhash(r['image_path']);min_dist=min((q^v).bit_count() for v in dev_h)
        if min_dist<=3:near.append({'path':r['relative_path'],'min_64bit_dhash_distance':min_dist})
    out={'n_test':len(rows),'class_counts':dict(Counter(CLASS_SHORT[r['label']] for r in rows)),
         'exact_duplicate_count':len(exact),'exact_duplicate_paths':exact,
         'near_duplicate_heuristic_count':len(near),'near_duplicate_candidates':near[:150],
         'warning':'dHash flags are heuristic, not evidence of shared patients; manually audit near matches.'}
    save_json(a.out/'holdout_overlap_audit.json',out)
    if exact:raise ValueError(f'{len(exact)} EXACT DEV-TEST duplicate images found. No test inference allowed.')
    return rows,out


def audit(a):
    a.out.mkdir(parents=True,exist_ok=True)
    a.features=a.root/'runs/adaptive_region_v3/region_features.npz'
    a.csv=a.root/'lung_adaptive_fusion_v1/anatomy_review_V3_eslestirilmis.csv'
    a.groups=a.root/'runs/chest_duplicate_aware_cv_v3/groups.csv'
    a.plan=a.root/'runs/chest_duplicate_aware_cv_v3/plan.json'
    a.data_root=a.root/'datasets/chest_ct/Data_exact_clean_v1_1'
    rel,y,grp,folds,X,fingerprint=load_data(a)
    s=inventory(a)
    s.update({'n_dev':len(y),'n_groups':int(len(set(grp))),
              'class_order':list(CLASS_SHORT),'plan_signature':SIGNATURE,
              'feature_fingerprint':fingerprint})
    # Only audit file names here, no need to open test for model selection.
    a.out.mkdir(parents=True,exist_ok=True)
    save_json(a.out/'audit_v36.json',s)
    return rel,y,grp,folds,X,fingerprint


def fit_expert(X,y,indices):
    sc=StandardScaler().fit(X[indices]);xx=np.clip(sc.transform(X[indices]),-8,8)
    mod=LogisticRegression(C=.03,class_weight='balanced',max_iter=1700)
    mod.fit(xx,y[indices])
    if list(mod.classes_) != [0,1,2,3]:raise ValueError('Selector class order mismatch')
    return (sc,mod)


def expert_predict(expert,X):
    sc,mod=expert
    return mod.predict_proba(np.clip(sc.transform(X),-8,8))


def selector_train(a,X,y,group,fingerprint):
    p=a.out/'train_view_choices.npz'
    sig=sha_text({'signature':SIGNATURE,'fingerprint':fingerprint,'split':'stratified-group-3',
                  'random_state':2626,'expert':'C.03_balanced','views':list(VIEW_NAMES[1:])})
    if p.exists():
        z=np.load(p,allow_pickle=False)
        if str(z['sig'])!=sig:raise ValueError('Selector cache mismatch')
        chosen=z['chosen'].astype(int)
        if chosen.shape!=(len(y),) or np.any((chosen<1)|(chosen>6)):raise ValueError('Selector cache invalid')
    else:
        inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2626)
        chosen=np.full(len(y),-1,dtype=int)
        used=np.zeros(len(y),dtype=bool)
        for tr,ho in inner.split(X[:,0],y,group):
            if set(group[tr]) & set(group[ho]):raise ValueError('Inner selection group leakage')
            probs=np.stack([expert_predict(fit_expert(X[:,j],y,tr),X[ho,j]) for j in range(1,7)],axis=1)
            chosen[ho]=1+probs.max(axis=2).argmax(axis=1)
            if used[ho].any():raise ValueError('Duplicate inner OOF samples')
            used[ho]=True
        if not used.all():raise ValueError('Incomplete inner OOF selector')
        np.savez_compressed(p,chosen=chosen,sig=sig)
    import joblib
    models_path=a.out/'selector_full_DEV.joblib'
    experts=[fit_expert(X[:,j],y,np.arange(len(y))) for j in range(1,7)]
    # This serialized sklearn model is produced locally, never load untrusted files.
    if models_path.exists():
        saved=joblib.load(models_path)
        if saved.get('fingerprint')!=fingerprint:raise ValueError('Full DEV selector does not match current DEV data')
    else:
        joblib.dump({'fingerprint':fingerprint,'experts':experts,
                     'model_id':MODEL_ID,'signature':SIGNATURE},models_path)
    return chosen


def make_transform(train):
    from torchvision import transforms as T
    t=[T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC)]
    if train:t.append(T.RandomAffine(degrees=7,translate=(.03,.03),scale=(.97,1.03),fill=0))
    t += [T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]
    return T.Compose(t)


def train_seed(a, rel, y, chosen, fingerprint, seed):
    import torch, timm
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.cpu:raise RuntimeError('CUDA not available (pass --cpu to explicitly permit CPU)')
    ck=a.out/f'resume_final_seed{seed}.pt';target=a.out/f'final_seed{seed}.pt'
    sig=sha_text({'version':VERSION,'fp':fingerprint,'seed':seed,'epochs':20,
        'batch':a.batch,'accum':a.accum,'backbone_lr':3e-5,'head_lr':3e-4,
        'global_loss':.8,'freeze_epochs':1,'model':MODEL_ID,'grad_ckpt':a.grad_ckpt})
    if target.exists():
        dd=torch.load(target,map_location='cpu',weights_only=True)
        if dd['run_sig']!=sig:raise ValueError(f'Existing seed model incompatible: {target}')
        print(f'ALREADY TRAINED final seed {seed}',flush=True);return
    random.seed(seed*1000);np.random.seed(seed*1000);torch.manual_seed(seed*1000)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed*1000)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    t=make_transform(True)
    class DS(Dataset):
        def __len__(self):return len(y)
        def __getitem__(self,i):
            im=img_read(a.data_root/rel[i])
            return t(im),t(crop(im,VIEW_NAMES[int(chosen[i])])),int(y[i])
    model=timm.create_model(MODEL_ID,pretrained=(not ck.exists()),num_classes=4).to(device)
    if a.grad_ckpt and hasattr(model,'set_grad_checkpointing'):model.set_grad_checkpointing(True)
    head=list(model.get_classifier().parameters());hi={id(x) for x in head}
    back=[x for x in model.parameters() if id(x) not in hi]
    opt=torch.optim.AdamW([{'params':back,'lr':3e-5},{'params':head,'lr':3e-4}],weight_decay=.02)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=1e-7)
    amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    counts=np.bincount(y,minlength=4)
    cw=torch.tensor(np.sqrt(len(y)/(4*counts)),device=device,dtype=torch.float32)
    start=1
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=False)
        if state['run_sig']!=sig:raise ValueError('Resume run signature mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['opt']);sch.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['rng_py']);np.random.set_state(state['rng_np']);torch.set_rng_state(state['rng_torch'])
        if amp:torch.cuda.set_rng_state_all(state['rng_cuda'])
        start=int(state['epoch'])+1
        print(f'RESUME seed={seed} from epoch {start}',flush=True)
    for ep in range(start,21):
        for p in back:p.requires_grad_(ep>1)
        model.train();opt.zero_grad(set_to_none=True)
        dl=DataLoader(DS(),batch_size=a.batch,shuffle=True,num_workers=0,pin_memory=amp,
            generator=torch.Generator().manual_seed(seed*2000+ep))
        running=0.0
        for bi,(g,l,yy) in enumerate(dl):
            g=g.to(device,non_blocking=True);l=l.to(device,non_blocking=True);yy=yy.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                gl,ll=model(torch.cat([g,l],0)).chunk(2,0)
                loss=.8*F.cross_entropy(gl,yy,weight=cw)+.2*F.cross_entropy(ll,yy,weight=cw)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at seed{seed} epoch{ep}')
            denom=min(a.accum,len(dl)-(bi//a.accum)*a.accum)
            scaler.scale(loss/denom).backward();running+=float(loss.detach())*len(yy)
            if (bi+1)%a.accum==0 or bi+1==len(dl):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        sch.step()
        print(f'FINAL REFIT seed={seed} epoch={ep}/20 loss={running/len(y):.5f}',flush=True)
        state={'run_sig':sig,'epoch':ep,'model':model.state_dict(),'opt':opt.state_dict(),
               'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),
               'rng_py':random.getstate(),'rng_np':np.random.get_state(),'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if amp else None}
        tmp=ck.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(ck)
    model_sd={k:v.detach().cpu() for k,v in model.state_dict().items()}
    output={'model':model_sd,'run_sig':sig,'seed':seed,'epochs':20,'class_order':list(CLASS_SHORT),
            'signature':SIGNATURE,'fingerprint':fingerprint,'view_names':list(VIEW_NAMES),
            'training_images':len(y),'algorithm':'V3.4 dual-view final refit on full 558 DEV'}
    tmp=target.with_suffix('.pt.tmp');torch.save(output,tmp);tmp.replace(target)
    ck.unlink(missing_ok=True)
    if amp:torch.cuda.empty_cache()


def frozen_features(a, test_rows, rel, dev_X):
    """Extract test image multi-view features, optionally verify original frozen cache on DEV."""
    import torch,timm
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.cpu:raise RuntimeError('CUDA required for extraction (or --cpu)')
    p=a.out/'holdout_frozen_features.npz'
    key=sha_text({'paths':[(r['relative_path'],r['sha256']) for r in test_rows],
                  'model':MODEL_ID,'views':VIEW_NAMES,'version':VERSION})
    if p.exists():
        z=np.load(p,allow_pickle=False)
        if str(z['key'])!=key:raise ValueError('Test cache corresponds to different images')
        xx=np.asarray(z['features'],dtype=np.float32)
        if xx.shape!=(len(test_rows),7,768):raise ValueError('Cached test features wrong shape')
        return xx
    t=make_transform(False)
    model=timm.create_model(MODEL_ID,pretrained=True,num_classes=0).to(device).eval()
    if a.grad_ckpt and hasattr(model,'set_grad_checkpointing'):model.set_grad_checkpointing(False)
    def encoded(path):
        im=img_read(path)
        crops=[t(crop(im,v)) for v in VIEW_NAMES]
        with torch.inference_mode():
            parts=[]
            for i in range(0,7,a.forward_batch):
                x=torch.stack(crops[i:i+a.forward_batch]).to(device)
                with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                    ff=model(x)
                parts.append(ff.float().cpu().numpy())
        return np.concatenate(parts)
    # Compare feature extraction on DEV samples to original V3 frozen 768-d cache.
    checks=[]
    for i in [0,len(rel)//2,len(rel)-1]:
        ff=encoded(a.data_root/rel[i]);diff=np.abs(ff-dev_X[i]);checks.append({'dev_idx':i,'mae':float(diff.mean()),'max_abs':float(diff.max())})
    save_json(a.out/'feature_extractor_replay.json',{'checks':checks})
    if any(it['mae']>0.02 for it in checks):
        raise RuntimeError('Frozen pretrained feature extractor differs from V3 cache. STOP; do not perform test.')
    xx=np.empty((len(test_rows),7,768),dtype=np.float32)
    for i,r in enumerate(test_rows):
        xx[i]=encoded(r['image_path'])
        if (i+1)%25==0 or i+1==len(test_rows):print(f'Frozen test features {i+1}/{len(test_rows)}',flush=True)
    tmp=p.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,features=xx,key=key)
    tmp.replace(p)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return xx


def select_new(a, Xtest, fingerprint):
    import joblib
    st=joblib.load(a.out/'selector_full_DEV.joblib')
    if st['fingerprint']!=fingerprint:raise ValueError('Selector fingerprint mismatch')
    probs=np.stack([expert_predict(st['experts'][j-1],Xtest[:,j]) for j in range(1,7)],axis=1)
    return 1+probs.max(2).argmax(1)


def predict_final(a, rows, choices, fingerprint):
    import torch,timm
    from torch.utils.data import Dataset,DataLoader
    if len(rows)!=len(choices):raise ValueError('Choice alignment invalid')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.cpu:raise RuntimeError('No CUDA: use --cpu explicitly')
    t=make_transform(False)
    class DS(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            im=img_read(rows[i]['image_path'])
            return t(im),t(crop(im,VIEW_NAMES[int(choices[i])])),i
    loader=DataLoader(DS(),batch_size=a.batch,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    P=np.zeros((len(rows),4),dtype=np.float64)
    PG=np.zeros_like(P);PR=np.zeros_like(P)
    for seed in SEEDS:
        file=a.out/f'final_seed{seed}.pt'
        if not file.exists():raise FileNotFoundError(f'Missing final model {file}; run --stage refit')
        st=torch.load(file,map_location='cpu',weights_only=True)
        if st['signature']!=SIGNATURE or st['fingerprint']!=fingerprint or st['seed']!=seed or st['epochs']!=20:
            raise ValueError('Model metadata mismatch; blocked')
        rc=a.out/'final_refit_config.json'
        if not rc.exists():raise FileNotFoundError('Missing final refit configuration')
        cfg=json.loads(rc.read_text(encoding='utf-8'))
        expected_sig=sha_text({'version':VERSION,'fp':fingerprint,'seed':seed,'epochs':20,
            'batch':cfg['batch'],'accum':cfg['accum'],'backbone_lr':3e-5,
            'head_lr':3e-4,'global_loss':.8,'freeze_epochs':1,
            'model':MODEL_ID,'grad_ckpt':cfg['grad_ckpt']})
        if st['run_sig']!=expected_sig:raise ValueError('Trained weight configuration signature mismatch')
        model=timm.create_model(MODEL_ID,pretrained=False,num_classes=4).to(device)
        model.load_state_dict(st['model'],strict=True)
        model.eval(); cursor=0
        with torch.inference_mode():
            for x,r,ids in loader:
                ids=np.asarray(ids,dtype=int)
                if not np.array_equal(ids,np.arange(cursor,cursor+len(ids))):raise ValueError('Inference index order mismatch')
                with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                    logit=model(torch.cat([x.to(device),r.to(device)]))
                g,l=logit.float().chunk(2,0)
                pg=g.softmax(-1).cpu().numpy();pr=l.softmax(-1).cpu().numpy()
                PG[cursor:cursor+len(ids)]+=pg/3
                PR[cursor:cursor+len(ids)]+=pr/3
                cursor+=len(ids)
        if cursor!=len(rows):raise ValueError('Incomplete inference')
        print(f'Completed test inference seed={seed}',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    P=.85*PG+.15*PR
    if not np.isfinite(P).all() or not np.allclose(P.sum(1),1,atol=1e-3):raise ValueError('Invalid output probability arrays')
    return PG,PR,P


def wilson(k,n,z=1.959963984540054):
    if n<=0:raise ValueError('Cannot compute accuracy with zero samples')
    p=k/n
    cen=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(max(0,cen-half)),float(min(1,cen+half))


def score(y,P):
    pred=np.argmax(P,axis=1);right=int(np.sum(pred==y));n=len(y)
    lo,hi=wilson(right,n)
    return {'n':n,'correct':right,'accuracy':float(right/n),'wilson95_accuracy_ci':[float(lo),float(hi)],
            'macro_f1':float(f1_score(y,pred,average='macro',labels=list(range(4)),zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'confusion_true_rows':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),
            'per_class_recall':{CLASS_SHORT[j]:float(np.mean(pred[y==j]==j)) if np.any(y==j) else None for j in range(4)}}


def write_predictions(file,rows,chosen,G,R,P,extra=None):
    fields=['image_path','relative_path','true_class','selected_view','global_pred','regional_pred','final_pred']
    fields+=[f'global_p{k}' for k in range(4)]
    fields+=[f'region_p{k}' for k in range(4)]
    fields+=[f'final_p{k}' for k in range(4)]
    fields+=extra or []
    with Path(file).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):
            item={'image_path':r['image_path'],'relative_path':r['relative_path'],
                  'true_class':r.get('label',-1),'selected_view':VIEW_NAMES[int(chosen[i])],
                  'global_pred':int(np.argmax(G[i])),'regional_pred':int(np.argmax(R[i])),
                  'final_pred':int(np.argmax(P[i]))}
            item.update({f'global_p{k}':float(G[i,k]) for k in range(4)})
            item.update({f'region_p{k}':float(R[i,k]) for k in range(4)})
            item.update({f'final_p{k}':float(P[i,k]) for k in range(4)})
            for k in extra or []:item[k]=r.get(k,'')
            w.writerow(item)


def test(a):
    rel,y,grp,folds,X,fp=audit(a)
    rows,overlap=build_test_inventory(a,rel)
    if (a.out/'test_report_v36.json').exists():
        raise RuntimeError('Test report already exists. LOCKED: do not repeatedly inspect and tune on holdout.')
    expected={str(s):str(a.out/f'final_seed{s}.pt') for s in SEEDS}
    missing=[p for p in expected.values() if not Path(p).exists()]
    if missing:raise FileNotFoundError('Final refit models missing, run --stage refit: '+str(missing))
    state={'status':'TEST_LOCKED_BEFORE_INFERENCE','method':'0.85 global + 0.15 selected region',
        'n_expected':len(rows),'signature':SIGNATURE,'fingerprint':fp,
        'test_manifest_sha':sha_text([(r['relative_path'],r['sha256']) for r in rows]),
        'weights_sha256':{str(s):sha_file(expected[str(s)]) for s in SEEDS},
        'notes':'Test never used for model selection. Only once. Comparisons to DEV OOF are descriptive due to final refit.'}
    lock=a.out/'holdout_test_lock.json'
    if lock.exists():
        existing=json.loads(lock.read_text(encoding='utf-8'))
        if existing!=state:raise RuntimeError('Locked test metadata does not match. Aborting.')
    else:save_json(lock,state)
    xt=frozen_features(a,rows,rel,X)
    selected=select_new(a,xt,fp)
    G,R,P=predict_final(a,rows,selected,fp)
    yy=np.asarray([r['label'] for r in rows])
    report={'evidence':'UNTOUCHED held-out 288 images from same source dataset; not independent hospital/patient validation',
            'model':'3-seed DeiT3 final refit trained on all 558 DEV images; NOT original 15-model V3.4 CV ensemble',
            'freeze':state,'duplicate_audit':{'exact_matches':overlap['exact_duplicate_count'],
                     'near_duplicate_candidates':overlap['near_duplicate_heuristic_count']},
            'methods':{'global':score(yy,G),'region':score(yy,R),'fixed_15':score(yy,P)},
            'disclaimer':'No threshold or region mixing optimized on test. Test-level slice association and patient IDs unknown.'}
    write_predictions(a.out/'test_predictions_v36.csv',rows,selected,G,R,P)
    save_json(a.out/'test_report_v36.json',report)
    print(json.dumps({'test_global':report['methods']['global']['accuracy'],
            'test_region':report['methods']['region']['accuracy'],
            'test_fixed_15':report['methods']['fixed_15']['accuracy'],
            'n':len(rows)},indent=2))


def parse_external(a):
    if a.manifest is None:raise ValueError('External evaluation needs --manifest CSV')
    rr=read_csv(a.manifest)
    for col in ('image_path','patient_id','class_name','cohort'):
        if not rr or col not in rr[0]:raise ValueError(f'External manifest missing {col}')
    rows=[]
    for r in rr:
        key=r['class_name'].strip().upper()
        if key not in ('ADC','SCC'):raise ValueError('External conditional two-class validation accepts ADC/SCC ONLY, got '+repr(key))
        p=Path(r['image_path'])
        if not p.is_absolute():p=(a.manifest.parent/p).resolve()
        if not p.is_file():raise FileNotFoundError(str(p))
        if p.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.webp'):
            raise ValueError('Curated PNG/JPG required. DICOM must first be transformed with a separately documented preprocessing protocol.')
        pid=r['patient_id'].strip();cohort=r['cohort'].strip()
        if not pid or not cohort:raise ValueError('Missing patient_id/cohort')
        rows.append({'image_path':str(p),'relative_path':r['image_path'],
                     'patient_id':pid,'cohort':cohort,'class_name':key,
                     'label':0 if key=='ADC' else 3,'sha256':sha_file(p)})
    if len({(r['cohort'],r['patient_id'],r['class_name']) for r in rows})==0:raise ValueError('Empty external cohort')
    perpat={}
    for r in rows:
        key=(r['cohort'],r['patient_id'])
        if key in perpat and perpat[key]!=r['label']:raise ValueError('Conflicting labels within patient')
        perpat[key]=r['label']
    return rows


def external(a):
    # External images are NEVER used to adjust the 0.15 mixing weight or class mapping.
    import torch,timm
    rel,y,grp,folds,X,fp=audit(a)
    rows=parse_external(a)
    cohort=set(r['cohort'] for r in rows)
    if len(cohort)!=1:raise ValueError('Run one external cohort at a time. Do not pool unrelated datasets.')
    if not (a.out/'test_report_v36.json').exists():
        raise RuntimeError('Finish locked internal heldout TEST before any external evaluation, as per registered analysis order.')
    output=a.out/('external_'+next(iter(cohort)).replace('/','_').replace('\\','_'))
    output.mkdir(exist_ok=True)
    if (output/'report_external.json').exists():raise RuntimeError('External report locked; cannot retune and rerun in same output')
    dev_sha={sha_file(a.data_root/r) for r in rel}
    collision=[r['image_path'] for r in rows if r['sha256'] in dev_sha]
    if collision:raise RuntimeError('Exact DEV/external image overlap: '+str(collision[:5]))
    frozen_hash=sha_text([(r['image_path'],r['sha256'],r['patient_id'],r['class_name']) for r in rows])
    signature={'plan_signature':SIGNATURE,'method':'0.85global+0.15region',
         'cohort':next(iter(cohort)),'external_manifest_hash':frozen_hash,
         'n_images':len(rows),'n_patients':len({r['patient_id'] for r in rows})}
    save_json(output/'external_protocol_lock.json',signature)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.cpu:raise RuntimeError('External inference requires CUDA or --cpu')
    t=make_transform(False)
    model=timm.create_model(MODEL_ID,pretrained=True,num_classes=0).to(device).eval()
    # Exact same ImageNet frozen encoder must reproduce original DEV feature cache.
    dev_t=make_transform(False)
    with torch.inference_mode():
        for ii in [0,len(rel)//2,len(rel)-1]:
            im=img_read(a.data_root/rel[ii]);q=[]
            for j in range(0,7,a.forward_batch):
                inp=torch.stack([dev_t(crop(im,v)) for v in VIEW_NAMES[j:j+a.forward_batch]]).to(device)
                with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                    q.append(model(inp).float().cpu().numpy())
            if float(np.abs(np.concatenate(q)-X[ii]).mean())>.02:
                raise RuntimeError('External frozen feature extractor differs from training cache; STOP')
    import joblib
    sel=joblib.load(a.out/'selector_full_DEV.joblib')
    if sel.get('fingerprint')!=fp:raise ValueError('Selector fingerprint mismatch')
    feats=np.empty((len(rows),7,768),dtype=np.float32)
    with torch.inference_mode():
        for i,r in enumerate(rows):
            im=img_read(r['image_path']); crops=[t(crop(im,v)) for v in VIEW_NAMES]
            pp=[]
            for j in range(0,7,a.forward_batch):
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    p=model(torch.stack(crops[j:j+a.forward_batch]).to(device))
                pp.append(p.float().cpu().numpy())
            feats[i]=np.concatenate(pp)
            if (i+1)%25==0:print('External feature extraction',i+1,len(rows),flush=True)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    selected=select_new(a,feats,fp)
    G,R,P=predict_final(a,rows,selected,fp)
    write_predictions(output/'external_image_predictions.csv',rows,selected,G,R,P,['patient_id','cohort','class_name'])
    # Conditional ADC vs SCC uses original softmax p_ADC and p_SCC; full-four errors separately counted.
    fourway=np.argmax(P,axis=1)
    ybin=np.array([r['label']==3 for r in rows],dtype=int)
    scc=P[:,3]/np.maximum(1e-12,P[:,0]+P[:,3])
    by={}
    for i,r in enumerate(rows):by.setdefault(r['patient_id'],[]).append(i)
    py=[];pp=[];fourcorrect=[]
    for patient,ids in by.items():
        label=ybin[ids[0]]
        py.append(label);prob=P[ids].mean(0);pp.append(prob[3]/max(1e-12,prob[0]+prob[3]));fourcorrect.append(int(np.argmax(prob)==rows[ids[0]]['label']))
    py=np.array(py,dtype=int);pp=np.array(pp,dtype=float)
    def binary_eval(ytrue,score):
        pred=(score>=.5).astype(int);n=len(ytrue);k=int(np.sum(ytrue==pred))
        ci=wilson(k,n)
        out={'n':n,'accuracy':float(k/n),'correct':k,'CI95_wilson':list(map(float,ci)),
             'balanced_accuracy':float(balanced_accuracy_score(ytrue,pred)),
             'macro_f1':float(f1_score(ytrue,pred,average='macro',zero_division=0)),
             'confusion_true_rows':confusion_matrix(ytrue,pred,labels=[0,1]).tolist()}
        if len(np.unique(ytrue))==2:
            out['auroc']=float(roc_auc_score(ytrue,score));out['ap_scc']=float(average_precision_score(ytrue,score))
        else:out['warning']='Only one true class; AUROC and AP unavailable.'
        return out
    results={'evidence':'External patient-labelled, curated ADC/SCC only. NOT four-class external validation.',
       'cohort':next(iter(cohort)),'manifest_hash':frozen_hash,
       'n_images':len(rows),'n_patients':len(py),
       'conditional_adc_scc_image_level':binary_eval(ybin,scc),
       'conditional_adc_scc_patient_level':binary_eval(py,pp),
       'full_four_class_on_adc_scc_subset_image_accuracy':float(np.mean(fourway==np.array([r['label'] for r in rows]))),
       'full_four_class_on_adc_scc_subset_patient_accuracy':float(np.mean(fourcorrect)),
       'limitations':['Patient cohort and ADC/SCC labels must have been audited externally; code does not verify clinical provenance.',
          'No NORMAL/LCC cases; cannot claim full four-class external performance.',
          'DICOM conversion/window and slice selection must be frozen in a separate log; PNG/JPG only.',
          'Model originally trained on chest PNG crops, may fail on full DICOM CT slices; report shift and failures.',
          'Multiple slices/patient are correlated; primary metric patient level.']}
    save_json(output/'report_external.json',results)
    print(json.dumps({'cohort':results['cohort'],'image_acc':results['conditional_adc_scc_image_level']['accuracy'],
        'patient_acc':results['conditional_adc_scc_patient_level']['accuracy'],'patients':len(py)},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage',required=True,choices=['audit','refit','test','external'])
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--out',type=Path)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accum',type=int,default=4)
    p.add_argument('--forward-batch',type=int,default=4)
    p.add_argument('--grad-ckpt',action='store_true',default=True)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--manifest',type=Path,default=None)
    a=p.parse_args();a.root=a.root.resolve()
    a.out=(a.out or a.root/'runs/adaptive_region_v36_test_locked').resolve()
    if a.out in ((a.root/'runs/adaptive_region_v34_20ep_3seed').resolve(),
                 (a.root/'runs/adaptive_region_v35_30ep_3seed').resolve()):
        raise ValueError('Output must be a NEW directory, not V3.4 / V3.5')
    if a.batch<1 or a.accum<1 or a.forward_batch<1:raise ValueError('Invalid positive batch/accum')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='audit':
        rel,y,g,folds,X,fp=audit(a)
        report=inventory(a)
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if a.stage=='refit':
        rel,y,group,folds,X,fp=audit(a)
        chosen=selector_train(a,X,y,group,fp)
        config={'algorithm':VERSION,'origin':'V3.4 20-epoch architecture','plan_signature':SIGNATURE,
                'feature_fingerprint':fp,'n_dev':558,'seeds':list(SEEDS),'epochs':20,
                'backbone_lr':3e-5,'head_lr':3e-4,'batch':a.batch,'accum':a.accum,
                'weight_decay':.02,'augment':'resize224+randomaffine7deg3pct',
                'global_loss_weight':.8,'region_inference_weight':.15,
                'final_refit':'3 seeds on ALL 558 DEV images, different from 15-fold OOF ensemble',
                'grad_ckpt':a.grad_ckpt}
        cpath=a.out/'final_refit_config.json'
        if cpath.exists():
            if json.loads(cpath.read_text(encoding='utf-8'))!=config:
                raise ValueError('Refit configuration changed in same output directory. Choose new --out.')
        else:save_json(cpath,config)
        for s in SEEDS:train_seed(a,rel,y,chosen,fp,s)
        save_json(a.out/'refit_complete.json',{'status':'3_seed_final_refit_complete',
                    'model_weights':{str(s):sha_file(a.out/f'final_seed{s}.pt') for s in SEEDS},
                    'plan_signature':SIGNATURE,'n_dev':558,'epochs':20,
                    'note':'No test dataset read during refit.'})
        print('Final 3-seed refit complete. Holdout test can now be evaluated once.',flush=True)
        return
    if a.stage=='test':test(a);return
    if a.stage=='external':external(a);return

if __name__=='__main__':main()
