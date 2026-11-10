#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3.3 Anchor-Preserving Cross-Fitted Multi-View Region Learning.

Research code for 558 dev PNG/JPG samples. Never reads test images.
No claims of patient-level independence, lung segmentation, or clinical utility.

Stages: check, recover, train, evaluate, all. Run --help for arguments.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from scipy.stats import binomtest

SIGNATURE = '522710c76a60df73d6776ae8a05e75d46c943dbc006215bf81d045a49ed3dae2'
VIEW_NAMES = ('global','center72','center90','upper_left','upper_right','lower_left','lower_right')
LABELS = ('adenocarcinoma','large.cell.carcinoma','normal','squamous.cell.carcinoma')
CLASS_SHORT = ('ADC','LCC','NOR','SCC')
MODEL_ID = 'deit3_base_patch16_224.fb_in22k_ft_in1k'


def read_csv(p):
    with Path(p).open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def normalize(s): return str(s).replace('\\','/').lstrip('./')


def save_json(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
    q.replace(p)


def sha_text(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def validate_prob(p,name='probabilities'):
    p=np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or (p < -1e-6).any() or np.max(np.abs(p.sum(1)-1)) > 3e-3:
        raise ValueError(f'{name}: invalid 4-class probability array, shape={p.shape}')
    return p.astype(np.float64)


def load_data(a):
    records = [r for r in read_csv(a.csv) if normalize(r['relative_path']).split('/')[0].lower() in ('train','valid')]
    rel = [normalize(r['relative_path']) for r in records]
    if len(rel)!=558 or len(set(rel))!=558: raise ValueError(f'Expected 558 unique DEV paths; found {len(rel)}')
    lookup_label={k:i for i,k in enumerate(LABELS)}
    y=[]
    for r in records:
        name=r.get('class_folder','').split('_')[0].lower()
        if name not in lookup_label: raise ValueError(f'Unknown label folder: {r.get("class_folder")}')
        yy=lookup_label[name]
        if r.get('true_class','') not in ('',str(yy)):raise ValueError('Inconsistent CSV labels')
        y.append(yy)
    y=np.array(y,dtype=np.int64)
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    if plan.get('signature')!=SIGNATURE:raise ValueError('Wrong V3 plan signature')
    pos={r:i for i,r in enumerate(rel)}
    folds=[]
    for k, fold in enumerate(plan['heldout_rel']):
        try: ids=np.array([pos[normalize(s)] for s in fold],dtype=int)
        except KeyError as e:raise ValueError(f'Plan contains path absent from CSV: {e}') from e
        folds.append(ids)
    if len(folds)!=5 or sorted(np.concatenate(folds).tolist())!=list(range(558)):
        raise ValueError('Fold coverage invalid')
    grow=read_csv(a.groups)
    gmap={normalize(r['rel']):str(r['group']) for r in grow}
    try:group=np.array([gmap[r] for r in rel])
    except KeyError as e:raise ValueError(f'Missing similarity-group row: {e}') from e
    for fold,k_ho in enumerate(folds):
        tr=np.setdiff1d(np.arange(len(rel)),k_ho)
        if set(group[tr])&set(group[k_ho]):raise ValueError(f'Group leakage in outer fold {fold}')
    z=np.load(a.features, allow_pickle=False)
    if str(z['signature'])!=SIGNATURE or list(z['rel'])!=rel or list(z['views'])!=list(VIEW_NAMES):
        raise ValueError('Frozen feature cache metadata mismatch: must be original V3 features')
    X=np.asarray(z['features'],dtype=np.float32)
    if X.shape!=(558,7,768) or not np.isfinite(X).all():raise ValueError(f'Invalid frozen features {X.shape}')
    features_digest=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    fingerprint=sha_text({'signature':SIGNATURE,'rel':rel,'y':y.tolist(),'group':group.tolist(),'views':VIEW_NAMES,'features_sha256':features_digest})
    return rel,y,group,folds,X,fingerprint


def expert_prob(X,y,train,query):
    sc=StandardScaler().fit(X[train]); A=np.clip(sc.transform(X[train]),-8,8)
    model=LogisticRegression(C=.03, class_weight='balanced',max_iter=1700)
    model.fit(A,y[train]); B=np.clip(sc.transform(X[query]),-8,8)
    return model.predict_proba(B)


def select_views(X,y,group,folds,k,out,fingerprint):
    """For each outer fold, inner-OOF region choice on outer training, outer-train-only on holdout."""
    p=out / f'choice_fold{k}.npz'; outer=folds[k]
    tr=np.setdiff1d(np.arange(len(y)),outer)
    cache_sig=sha_text({'f':fingerprint,'k':k,'inner_seed':1137+k,'n_inner':3,'selector':'logregC003conf'})
    if p.exists():
        z=np.load(p,allow_pickle=False)
        if str(z['cache_sig'])!=cache_sig or not np.array_equal(z['train_idx'],tr) or not np.array_equal(z['held_idx'],outer):
            raise ValueError(f'Existing selector cache incompatible: {p}')
        cv=np.asarray(z['choice'],dtype=np.int16)
        if cv.shape!=(len(y),) or np.any(cv[tr]<1) or np.any(cv[outer]<1):raise ValueError('Incomplete selector cache')
        return cv
    inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=1137+k)
    choice=np.full(len(y),-1,dtype=np.int16)
    seen=np.zeros(len(y),dtype=bool)
    for inner_train,inner_valid in inner.split(X[tr,0],y[tr],group[tr]):
        train_idx=tr[inner_train];val_idx=tr[inner_valid]
        if set(group[train_idx])&set(group[val_idx]):raise ValueError('Inner fold group leakage')
        probs=np.stack([expert_prob(X[:,j],y,train_idx,val_idx) for j in range(1,7)],axis=1)
        choice[val_idx]=1+probs.max(2).argmax(1)
        if np.any(seen[val_idx]):raise ValueError('Inner index predicted twice')
        seen[val_idx]=True
    if not seen[tr].all():raise ValueError('Inner OOF selector incomplete')
    # Critical: heldout labels never appear in training selectors.
    q=np.stack([expert_prob(X[:,j],y,tr,outer) for j in range(1,7)],axis=1)
    choice[outer]=1+q.max(2).argmax(1)
    if (choice<1).any() or (choice>6).any():raise ValueError('Invalid crop choice')
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p, choice=choice,train_idx=tr,held_idx=outer,cache_sig=cache_sig)
    return choice


def crop(im,view):
    w,h=im.size
    if view=='global':return im
    f=.9 if view=='center90' else .72
    cw,ch=max(1,round(w*f)),max(1,round(h*f))
    if view.startswith('center'):x,y=(w-cw)//2,(h-ch)//2
    else:
        x=0 if view.endswith('left') else w-cw
        y=0 if view.startswith('upper') else h-ch
    return im.crop((x,y,x+cw,y+ch))


def train_one(a,rel,y,folds,chosen,k,s,fingerprint):
    import torch,timm
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from torchvision import transforms as T
    from torch.utils.data import DataLoader, Dataset
    if not torch.cuda.is_available() and not a.cpu:raise RuntimeError('CUDA not available. Use --cpu explicitly if intended')
    device=torch.device('cpu' if a.cpu else 'cuda')
    held=folds[k];train=np.setdiff1d(np.arange(len(y)),held)
    seed=s*1000+k
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    augment=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomAffine(degrees=7,translate=(.03,.03),scale=(.97,1.03),fill=0),
                      T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    plain=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC),
                     T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    class CTData(Dataset):
        def __init__(self,ids,is_train):self.ids=np.array(ids,dtype=int);self.t=augment if is_train else plain
        def __len__(self):return len(self.ids)
        def __getitem__(self,q):
            i=int(self.ids[q]);file=a.data_root/rel[i]
            with Image.open(file) as raw: im=ImageOps.exif_transpose(raw).convert('RGB')
            return self.t(im), self.t(crop(im,VIEW_NAMES[int(chosen[i])])),int(y[i]),i
    model=timm.create_model(MODEL_ID,pretrained=True,num_classes=4).to(device)
    if a.grad_ckpt and hasattr(model,'set_grad_checkpointing'):model.set_grad_checkpointing(True)
    head=list(model.get_classifier().parameters());hi={id(p) for p in head}
    back=[p for p in model.parameters() if id(p) not in hi]
    optimizer=torch.optim.AdamW([{'params':back,'lr':a.backbone_lr},
                                 {'params':head,'lr':a.head_lr}],weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-7)
    amp=device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    count=np.bincount(y[train],minlength=4)
    if (count==0).any():raise ValueError('Fold class missing')
    cw=torch.tensor(np.sqrt(len(train)/(4*count)),device=device,dtype=torch.float32)
    ck=a.out/f'resume_seed{s}_fold{k}.pt'
    config_sig=sha_text({'fingerprint':fingerprint,'fold':k,'seed':s,'epochs':a.epochs,'lr':a.backbone_lr,
                         'head_lr':a.head_lr,'batch':a.batch,'accum':a.accum,'grad_ckpt':a.grad_ckpt,
                         'freeze_epochs':a.freeze_epochs,'loss_global':a.global_loss})
    start=1
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=False)
        if state.get('config_sig')!=config_sig:raise ValueError('Resume checkpoint has incompatible config')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        if amp:torch.cuda.set_rng_state_all(state['rng_cuda'])
        start=int(state['epoch'])+1
        print(f'RESUMED seed{s} fold{k} from epoch {start-1}',flush=True)
    for ep in range(start,a.epochs+1):
        freeze=ep <= a.freeze_epochs
        for p in back:p.requires_grad_(not freeze)
        dl=DataLoader(CTData(train,True),batch_size=a.batch,shuffle=True,num_workers=0,
                      pin_memory=amp,generator=torch.Generator().manual_seed(seed*2000+ep))
        model.train();optimizer.zero_grad(set_to_none=True);running=0.;n=0
        for batch,(xx,rr,yy,_) in enumerate(dl):
            xx=xx.to(device,non_blocking=True);rr=rr.to(device,non_blocking=True)
            yy=yy.to(device,non_blocking=True)
            # Two branches share the identical DeiT3 weights; logits computed in one pass.
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(torch.cat([xx,rr],dim=0))
                logit_g,logit_r=logits.chunk(2,dim=0)
                loss=a.global_loss*F.cross_entropy(logit_g,yy,weight=cw)+(1-a.global_loss)*F.cross_entropy(logit_r,yy,weight=cw)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss, fold{k} ep{ep}')
            window=min(a.accum,len(dl)-(batch//a.accum)*a.accum)
            scaler.scale(loss/window).backward()
            running+=float(loss.detach())*len(yy);n+=len(yy)
            if (batch+1)%a.accum==0 or (batch+1)==len(dl):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        print(f'fold={k} seed={s} epoch={ep}/{a.epochs} loss={running/max(1,n):.5f}',flush=True)
        # Save AFTER every epoch; atomically, to avoid losing hours to an inference bug.
        state=dict(config_sig=config_sig,epoch=ep,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_python=random.getstate(),
                   rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
                   rng_cuda=torch.cuda.get_rng_state_all() if amp else None)
        ck_tmp=ck.with_suffix('.pt.tmp');torch.save(state,ck_tmp);ck_tmp.replace(ck)
    model.eval();pglobal=np.full((len(held),4),np.nan,dtype=np.float32)
    pregion=np.full((len(held),4),np.nan,dtype=np.float32)
    cursor=0
    with torch.inference_mode():
        for xx,rr,_,ids in DataLoader(CTData(held,False),batch_size=a.batch,shuffle=False,num_workers=0):
            inds=np.asarray(ids,dtype=int);stop=cursor+len(inds)
            if not np.array_equal(held[cursor:stop],inds):raise ValueError(f'Inference order mismatch at {cursor}:{stop}')
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(torch.cat([xx.to(device),rr.to(device)],dim=0))
            gl,lo=logits.float().chunk(2,dim=0)
            pglobal[cursor:stop]=gl.softmax(1).cpu().numpy()
            pregion[cursor:stop]=lo.softmax(1).cpu().numpy()
            cursor=stop
    if cursor!=len(held):raise ValueError('Missing heldout inference')
    validate_prob(pglobal,'new global');validate_prob(pregion,'new local')
    dest=a.out/f'pred_seed{s}_fold{k}.npz'
    tmp=dest.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,global_probs=pglobal,region_probs=pregion,idx=held,seed=s,fold=k,
                            signature=SIGNATURE,config_sig=config_sig,selected_views=chosen[held])
    tmp.replace(dest)
    if a.keep_final_weights:
        torch.save({'model':model.state_dict(),'signature':SIGNATURE,'seed':s,'fold':k,
                    'classes':LABELS,'selected_views_strategy':'crossfitted training selector'},a.out/f'weights_seed{s}_fold{k}.pt')
    ck.unlink(missing_ok=True)
    del model,optimizer
    if amp:torch.cuda.empty_cache()


def load_baseline_csv(file,rel,y):
    entries=read_csv(file)
    need={'relative_path','true_class','p0','p1','p2','p3','plan_signature'}
    if not entries or not need.issubset(entries[0]):raise ValueError('Baseline CSV needs '+','.join(sorted(need)))
    lookup={normalize(r['relative_path']):r for r in entries}
    if len(lookup)!=len(rel) or set(lookup)!=set(rel):raise ValueError('Baseline CSV does not match 558 paths')
    if any(lookup[p]['plan_signature']!=SIGNATURE for p in rel):raise ValueError('Baseline plan signature mismatch')
    if any(int(lookup[p]['true_class'])!=int(y[i]) for i,p in enumerate(rel)):raise ValueError('Baseline true-class mismatch')
    return validate_prob(np.array([[float(lookup[p][f'p{j}']) for j in range(4)] for p in rel]),'baseline')


def reconstruct_baseline(a,rel,y,folds):
    """STRICT: 3 seeds * 5 folds, matching heldout idx and signature and 534/558.
    Ambiguous or incomplete reconstruction is rejected, never quietly substituted.
    """
    if a.baseline_oof and not a.baseline_oof.exists():raise FileNotFoundError(f'Explicit baseline OOF missing: {a.baseline_oof}')
    if a.baseline_oof and a.baseline_oof.exists():
        P=load_baseline_csv(a.baseline_oof,rel,y)
        if int(np.sum(P.argmax(1)==y))!=534:
            raise ValueError('Baseline CSV does not reproduce historical 534/558; wrong artifact?')
        return P,'supplied_csv'
    built=a.out/'original_v3_recovered_oof.csv'
    if built.exists():
        P=load_baseline_csv(built,rel,y)
        if int(np.sum(P.argmax(1)==y))==534:return P,'previously_reconstructed'
        raise ValueError('Recovered baseline changed, aborting')
    root=a.baseline_root
    if root is None or not root.is_dir():return None,'no baseline root'
    hits=[]
    for p in root.rglob('*.npz'):
        if not re.search(r'seed\d+_fold\d+',p.name,re.I):continue
        hits.append(p)
    candidates=[];by_dir=defaultdict(list)
    for p in hits:
        try:
            z=np.load(p,allow_pickle=False)
            fields=set(z.files)
            probkey=next((q for q in ('probs','probabilities','p') if q in fields),None)
            idxkey=next((q for q in ('idx','indices','held_idx','heldout_idx') if q in fields),None)
            if not probkey or not idxkey:continue
            match=re.search(r'seed(\d+)_fold(\d+)',p.name,re.I)
            s,k=int(match.group(1)),int(match.group(2))
            if s not in (42,52,62) or k not in range(5):continue
            if 'plan_signature' not in fields or str(z['plan_signature'])!=SIGNATURE:continue
            idx=np.asarray(z[idxkey],dtype=int);v=validate_prob(z[probkey],str(p))
            if not np.array_equal(idx,folds[k]) or v.shape!=(len(idx),4):continue
            by_dir[p.parent].append((s,k,idx,v,p))
            candidates.append({'file':str(p),'seed':s,'fold':k,'status':'valid'})
        except (ValueError,OSError,KeyError,RuntimeError) as e:
            candidates.append({'file':str(p),'status':f'rejected: {type(e).__name__}'})
    if candidates:
        with (a.out/'baseline_scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['file','seed','fold','status'],extrasaction='ignore');w.writeheader();w.writerows(candidates)
    results=[]
    for folder,items in by_dir.items():
        dic={(s,k):(idx,v) for s,k,idx,v,_ in items}
        if len(dic)!=15:continue
        per=[]
        for s in (42,52,62):
            pp=np.full((len(rel),4),np.nan)
            for k in range(5):
                idx,v=dic[(s,k)];pp[idx]=v
            if not np.isfinite(pp).all():break
            per.append(pp)
        if len(per)==3:
            P=validate_prob(np.mean(per,axis=0),'reconstructed baseline')
            if int(np.sum(P.argmax(1)==y))==534:results.append((folder,P))
    if len(results)!=1:
        msg=f'Found {len(results)} uniquely reproducible baseline directories (expected exactly one).'
        (a.out/'baseline_status.txt').write_text(msg+'\nNo baseline blending performed.\n',encoding='utf-8')
        return None,msg
    folder,P=results[0]
    with built.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['relative_path','true_class','p0','p1','p2','p3','plan_signature'])
        for p,yy,row in zip(rel,y,P):w.writerow([p,int(yy),*map(float,row),SIGNATURE])
    return P,str(folder)


def calc_metric(y,P):
    pred=P.argmax(1)
    return {'acc':float(accuracy_score(y,pred)),'correct':int((pred==y).sum()),
            'balanced_acc':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'recall':{CLASS_SHORT[i]:float(np.mean(pred[y==i]==i)) for i in range(4)},
            'confusion':confusion_matrix(y,pred,labels=list(range(4))).tolist()}


def compare(y,groups,B,P,bootstrap=2000):
    b=B.argmax(1)==y;c=P.argmax(1)==y
    rescued=int(np.sum(~b&c));lost=int(np.sum(b&~c));n=rescued+lost
    rng=np.random.default_rng(2606);uniq=np.unique(groups)
    members=[np.where(groups==g)[0] for g in uniq]
    diff=[]
    for _ in range(bootstrap):
        picks=rng.integers(len(uniq),size=len(uniq))
        ids=np.concatenate([members[j] for j in picks])
        diff.append(100*(c[ids].mean()-b[ids].mean()))
    return {'rescued':rescued,'lost':lost,'delta_pp':100*float(c.mean()-b.mean()),
            'group_bootstrap_95ci_pp':np.percentile(diff,[2.5,97.5]).tolist(),
            'image_level_mcnemar_exploratory_p':float(binomtest(min(lost,rescued),n,.5).pvalue if n else 1.)}


def assemble(a,rel,y,groups,folds,chosen):
    outputs={name:np.full((len(rel),4),np.nan) for name in ('trained_global','trained_region')}
    for s in a.seeds:
        temp={n:np.full((len(rel),4),np.nan) for n in outputs}
        for k,ho in enumerate(folds):
            f=a.out/f'pred_seed{s}_fold{k}.npz'
            if not f.exists():raise FileNotFoundError('Incomplete experiment: '+str(f))
            z=np.load(f,allow_pickle=False)
            if str(z['signature'])!=SIGNATURE or int(z['seed'])!=s or int(z['fold'])!=k or not np.array_equal(z['idx'],ho):
                raise ValueError(f'Invalid fold cache: {f}')
            if not np.array_equal(z['selected_views'],chosen[k][ho]):raise ValueError('Heldout crop selections changed')
            temp['trained_global'][ho]=validate_prob(z['global_probs'])
            temp['trained_region'][ho]=validate_prob(z['region_probs'])
        for n in outputs:
            if not np.isfinite(temp[n]).all():raise ValueError('Incomplete OOF')
            if np.isnan(outputs[n]).all():outputs[n]=temp[n].copy()
            else:outputs[n]+=temp[n]
    for n in outputs:outputs[n]=validate_prob(outputs[n]/len(a.seeds))
    G,R=outputs['trained_global'],outputs['trained_region']
    outputs['internal_fixed_15']=validate_prob(.85*G+.15*R)
    # Independently predeclared confidence rule. Never uses true classes of the evaluated sample.
    conf=G.max(1);local_conf=R.max(1)
    wg=.15*np.clip((.90-conf)/.35,0,1)*np.clip((local_conf-.55)/.35,0,1)
    outputs['internal_guarded']=validate_prob((1-wg[:,None])*G+wg[:,None]*R)
    B,source=reconstruct_baseline(a,rel,y,folds)
    if B is not None:
        outputs['original_v3_ensemble']=B
        outputs['anchor_fixed_05']=validate_prob(.95*B+.05*R)
        outputs['anchor_fixed_10']=validate_prob(.90*B+.10*R)
        # Conservative anchored preset, no automatic threshold fitting on OOF labels.
        gate=.15*np.clip((.90-B.max(1))/.35,0,1)*np.clip((R.max(1)-.55)/.35,0,1)
        outputs['anchor_guarded_15']=validate_prob((1-gate[:,None])*B+gate[:,None]*R)
    return outputs,source


def evaluate(a,rel,y,groups,folds,chosen):
    outputs,source=assemble(a,rel,y,groups,folds,chosen)
    report={'protocol':'V3.3 Cross-Fitted Dual-View DeiT3 + Baseline-Anchored Rescue',
            'evidence':'558 DEV images, 508 similarity groups; NO patient-disjoint/external validation',
            'signature':SIGNATURE,'seeds':a.seeds,'baseline_source':source,
            'hyperparameters_predeclared':{'global_training_loss_weight':a.global_loss,
             'internal_fixed_region_weight':.15,'anchor_weights':[.05,.10],
             'guarded_max_weight':.15,'guarded_baseline_confidence_threshold':.90,
             'guarded_regional_confidence_threshold':.55},
            'methods':{n:calc_metric(y,p) for n,p in outputs.items()},'comparisons':{},
            'limitations':['These are geometric crops, not validated lung segmentation.',
              'These are similarity-group heldouts, NOT patient-disjoint subjects.',
              'Prior V3/V3.1/V3.2 development guided design: this is exploratory, NOT untouched confirmation.',
              'V3.3 fixed/gated blending is not proof of calibrated confidence or clinical reliability.',
              'Unknown slice source overlap remains; independent patient-identified cohort is needed.',
              'An old V3 OOF file is used only if class order, 558 paths, plan signature and 534/558 reproduce.']}
    B=outputs.get('original_v3_ensemble')
    if B is not None:
        for name,P in outputs.items():
            if name!='original_v3_ensemble':report['comparisons'][name+'_vs_original_v3']=compare(y,groups,B,P,a.bootstrap)
    else:
        report['warning']='Old V3 ensemble OOF NOT RECONSTRUCTED; no legitimate comparison to 95.70% yet.'
    save_json(a.out/'report_v33.json',report)
    foldmap=np.full(len(rel),-1)
    for k,ho in enumerate(folds):foldmap[ho]=k
    cols=['relative_path','true_class','group','outer_fold','selected_region']
    for name in outputs:
        cols.extend([name+'_pred']+[name+f'_p{j}' for j in range(4)])
    with (a.out/'oof_v33.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for i,path in enumerate(rel):
            k=int(foldmap[i]);v=int(chosen[k][i]);row={'relative_path':path,'true_class':int(y[i]),
                 'group':str(groups[i]),'outer_fold':k,'selected_region':VIEW_NAMES[v]}
            for name,P in outputs.items():
                row[name+'_pred']=int(np.argmax(P[i]))
                for j in range(4):row[name+f'_p{j}']=float(P[i,j])
            w.writerow(row)
    print(json.dumps({'baseline_source':source,'methods':{n:{'accuracy':v['acc'], 'correct':v['correct'], 'macro_f1':v['macro_f1']} for n,v in report['methods'].items()},
                      'comparisons':report['comparisons']},ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage',choices=['check','recover','train','evaluate','all'],default='check')
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--csv',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--groups',type=Path)
    p.add_argument('--features',type=Path);p.add_argument('--data-root',type=Path)
    p.add_argument('--baseline-root',type=Path);p.add_argument('--baseline-oof',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accum',type=int,default=4)
    p.add_argument('--backbone-lr',type=float,default=3e-5);p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--freeze-epochs',type=int,default=1);p.add_argument('--global-loss',type=float,default=.8)
    p.add_argument('--cpu',action='store_true');p.add_argument('--grad-ckpt',action='store_true')
    p.add_argument('--keep-final-weights',action='store_true')
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();a.root=a.root.resolve()
    defaults={'csv':'lung_adaptive_fusion_v1/anatomy_review_V3_eslestirilmis.csv',
              'plan':'runs/chest_duplicate_aware_cv_v3/plan.json',
              'groups':'runs/chest_duplicate_aware_cv_v3/groups.csv',
              'features':'runs/adaptive_region_v3/region_features.npz',
              'data_root':'datasets/chest_ct/Data_exact_clean_v1_1',
              'baseline_root':'runs/chest_duplicate_aware_cv_v3',
              'out':'runs/adaptive_region_v33'}
    for key,val in defaults.items():
        curr=getattr(a,key)
        if curr is None:setattr(a,key,a.root/val)
        elif not curr.is_absolute():setattr(a,key,a.root/curr)
    if a.baseline_oof and not a.baseline_oof.is_absolute():a.baseline_oof=a.root/a.baseline_oof
    if not 0 < a.global_loss < 1:raise ValueError('--global-loss must be within (0,1)')
    if a.batch < 1 or a.accum < 1 or a.epochs < 1:raise ValueError('Bad training settings')
    a.out.mkdir(parents=True,exist_ok=True)
    rel,y,groups,folds,X,fingerprint=load_data(a)
    train_config={'signature':SIGNATURE,'fingerprint':fingerprint,'epochs':a.epochs,
       'batch':a.batch,'accum':a.accum,'backbone_lr':a.backbone_lr,
       'head_lr':a.head_lr,'freeze_epochs':a.freeze_epochs,
       'global_loss':a.global_loss,'grad_ckpt':a.grad_ckpt,'model':MODEL_ID}
    config_file=a.out/'training_config_v33.json'
    # Pure preflight/recover must not lock GPU, batch or training settings.
    if a.stage in ('train','all','evaluate'):
        if config_file.exists():
            if json.loads(config_file.read_text(encoding='utf-8')) != train_config:
                raise ValueError('V3.3 run configuration differs from existing output; choose a NEW --out directory')
        elif a.stage!='evaluate':
            save_json(config_file,train_config)
        else:
            raise FileNotFoundError('No prior V3.3 training configuration exists in output directory')
    # Fail fast when requested input is absent; NEVER silently search test or reconstruct fake data.
    if a.stage in ('train','all'):
        missing=[str(a.data_root/path) for path in rel if not (a.data_root/path).is_file()]
        if missing:raise FileNotFoundError(f'Missing {len(missing)} DEV images, e.g. {missing[0]}')
    print(f'FROZEN PLAN OK {SIGNATURE[:20]}... | DEV={len(rel)} | groups={len(np.unique(groups))} | label counts={dict(zip(CLASS_SHORT,np.bincount(y)))}',flush=True)
    choices=[select_views(X,y,groups,folds,k,a.out,fingerprint) for k in range(5)]
    # Verify each training choice is inner OOF; heldout choice is fitted on all outer train only.
    selector_info={'signature':SIGNATURE,'n_dev':len(y),'n_similarity_groups':len(np.unique(groups)),
                   'train_region_choice':'3-fold inner OOF on each outer train',
                   'heldout_region_choice':'train-on-outer-train only',
                   'selected_view_counts_per_outer_fold':[
                       {VIEW_NAMES[j]:int(np.sum(c[folds[k]]==j)) for j in range(1,7)} for k,c in enumerate(choices)]}
    save_json(a.out/'preflight_v33.json',selector_info)
    if a.stage=='check':
        B,src=reconstruct_baseline(a,rel,y,folds)
        print(f'PREFLIGHT OK; baseline={src}; reconstructed={B is not None}; no training run',flush=True)
        return
    if a.stage=='recover':
        B,src=reconstruct_baseline(a,rel,y,folds)
        print(f'BASELINE status: {src}; recovered={B is not None}',flush=True)
        return
    if a.stage in ('train','all'):
        for s in a.seeds:
            for k,ho in enumerate(folds):
                dest=a.out/f'pred_seed{s}_fold{k}.npz'
                if dest.exists():
                    z=np.load(dest,allow_pickle=False)
                    if int(z['seed'])!=s or int(z['fold'])!=k or str(z['signature'])!=SIGNATURE or not np.array_equal(z['idx'],ho):
                        raise ValueError(f'Existing prediction cache incompatible: {dest}')
                    validate_prob(z['global_probs']);validate_prob(z['region_probs'])
                    if not np.array_equal(z['selected_views'],choices[k][ho]):raise ValueError('Selector differs from trained model cache')
                    print(f'SKIP already completed fold{k} seed{s}',flush=True);continue
                print(f'TRAIN seed{s} fold{k} train={len(y)-len(ho)} holdout={len(ho)}',flush=True)
                train_one(a,rel,y,folds,choices[k],k,s,fingerprint)
    if a.stage in ('evaluate','all','train'):
        evaluate(a,rel,y,groups,folds,choices)

if __name__=='__main__':main()
