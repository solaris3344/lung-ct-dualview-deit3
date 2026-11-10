#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duplicate-aware, review-first 558-DEV grouping. No TEST folder read.

Run: python duplicate_review.py --stage audit|finalize --data ... --out ...
Only exact pixel identity auto-links; all approximate image/filename links require review.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from devcv import dev_rows, load_features, fingerprint, save_json

AUDIT_VERSION = 'CHEST_DUPLICATE_AWARE_V3'

def sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def copy_root(name):
    stem = Path(name).stem.lower().strip()
    if not re.search(r'(?i)(?:\bcopy\b|\bkopya\b)', stem):
        return None
    stem = re.sub(r'\s*[-_ ]*\b(?:copy|kopya)\b\s*', ' ', stem, flags=re.I)
    stem = re.sub(r'\s*\(\s*\d+\s*\)\s*', ' ', stem)
    stem = re.sub(r'\s*[-_ ]+\s*', ' ', stem).strip()
    return stem or None

def corr_gray(a, b):
    def vec(p):
        with Image.open(p) as raw:
            im = ImageOps.exif_transpose(raw).convert('L').resize((96,96), Image.Resampling.BICUBIC)
        x = np.asarray(im, dtype=np.float32).ravel()
        x -= x.mean(); x /= np.linalg.norm(x) + 1e-10
        return x
    return float(vec(a) @ vec(b))

def tiny_card(impath, title, rows, root, members, w=164, max_show=40):
    members = list(members)[:max_show]
    ncol = 4
    nrow = (len(members)+ncol-1)//ncol
    dest = Image.new('RGB', (ncol*(w+12)+6, 29+nrow*(w+48)), (246,248,251))
    d = ImageDraw.Draw(dest); d.text((8,8), title[:85], fill=(0,0,0))
    for k,i in enumerate(members):
        y = 29+(k//ncol)*(w+48); x=8+(k%ncol)*(w+12)
        with Image.open(rows[i]['path']) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
        im.thumbnail((w,w))
        canvas = Image.new('RGB',(w,w),'black'); canvas.paste(im,((w-im.width)//2,(w-im.height)//2))
        dest.paste(canvas,(x,y))
        r=rows[i]; d.text((x,y+w+2), (r['label'][:8]+' '+r['split']), fill=(0,0,0))
        d.text((x,y+w+18), Path(r['path']).name[:24],fill=(0,0,0))
    impath.parent.mkdir(parents=True, exist_ok=True); dest.save(impath)

def audit(a):
    if (a.out/'plan.json').exists() or (a.out/'approved_links.json').exists():
        raise RuntimeError('Audit/approval frozen; use new --out folder to change candidate protocol')
    dev, _ = dev_rows(a.data)
    fm, feat_model = load_features(a, dev)
    if len(dev)!=len(fm): raise RuntimeError('Feature count mismatch')
    s = np.clip(fm@fm.T, -1, 1)
    dh = np.array([int(r['dhash64'],16) for r in dev],dtype=np.uint64)
    family_map=defaultdict(list)
    for i,r in enumerate(dev):
        root=copy_root(Path(r['path']).name)
        if root:
            family_map[(r['label'],root)].append(i)
    families=[]
    for (label,root),inds in sorted(family_map.items()):
        if len(inds)<2: continue
        rels=sorted(dev[i]['rel'] for i in inds)
        fid=sha(['family',label,root,rels])[:16]
        families.append({'family_id':fid,'label':label,'copy_root':root,'n':len(inds),
                         'spans_orig_train_valid':int(len({dev[i]['split'] for i in inds})>1),
                         'members': '|'.join(rels),'decision':'review'})
        tiny_card(a.out/'sheets'/'families'/f'{fid}.jpg',f'{label} / {root} / {len(inds)}',dev,root,inds)
    # Pair proposals: all dHash <= threshold and all ImageNet cosine >= cutoff.
    # These are *NOT* automatic links, and exact pixel identities already mandatory.
    iu, ju = np.triu_indices(len(dev), 1)
    candidates=[]
    for i,j in zip(iu,ju):
        i=int(i); j=int(j)
        h=int((int(dh[i])^int(dh[j])).bit_count())
        cs=float(s[i,j])
        if dev[i]['pixel_sha256']==dev[j]['pixel_sha256']: continue
        reason=[]
        if h<=a.max_hamming: reason.append('dhash')
        if cs>=a.min_cosine: reason.append('imagenet')
        if not reason: continue
        ri,rj=dev[i]['rel'],dev[j]['rel']
        pid=sha(['pair',ri,rj])[:16]
        candidates.append({'pair_id':pid,'rel_a':ri,'rel_b':rj,
                           'label_a':dev[i]['label'],'label_b':dev[j]['label'],
                           'hamming':h,'imagenet_cosine':f'{cs:.6f}',
                           'pixel_corr96':f'{corr_gray(dev[i]["path"],dev[j]["path"]):.5f}',
                           'reason':'+'.join(reason),'decision':'review'})
    candidates.sort(key=lambda p:(p['label_a']!=p['label_b'],
                                  -float(p['pixel_corr96']),-float(p['imagenet_cosine'])))
    for page,st in enumerate(range(0, min(len(candidates),a.sheet_limit),4)):
        inds=[]
        for c in candidates[st:st+4]:
            for rel in (c['rel_a'],c['rel_b']):
                inds.append(next(i for i,r in enumerate(dev) if r['rel']==rel))
        tiny_card(a.out/'sheets'/'pair_candidates'/f'page_{page:03d}.jpg',
                  f'Potential duplicates: {st+1}-{st+len(inds)//2}',dev,'',inds)
    famfields=['family_id','label','copy_root','n','spans_orig_train_valid','members','decision']
    pairfields=['pair_id','rel_a','rel_b','label_a','label_b','hamming','imagenet_cosine','pixel_corr96','reason','decision']
    write_csv(a.out/'family_decisions.csv', famfields, families)
    write_csv(a.out/'pair_decisions.csv', pairfields, candidates)
    spec={'version':AUDIT_VERSION,'fingerprint':fingerprint(dev),'feat_model':feat_model,
          'feature_sha256':hashlib.sha256((a.out/'dev_features.npz').read_bytes()).hexdigest(),
          'params':{'max_hamming':a.max_hamming,'min_cosine':a.min_cosine},
          'families':{x['family_id']: {'label':x['label'],'members':x['members']} for x in families},
          'pairs':{x['pair_id']: [x['rel_a'],x['rel_b']] for x in candidates}}
    save_json(a.out/'duplicate_candidates.json',spec)
    report={'version':AUDIT_VERSION,'dev_count':len(dev),'copy_family_candidates':len(families),
            'pair_candidates':len(candidates),
            'pair_cross_label':sum(p['label_a']!=p['label_b'] for p in candidates),
            'note':'Exact pixel links mandatory; every approximate link requires human approval. NO test access.'}
    save_json(a.out/'duplicate_audit_summary.json',report)
    print('DUPLICATE AUDIT:',json.dumps(report,ensure_ascii=False))
    print('Review:',a.out/'family_decisions.csv',a.out/'pair_decisions.csv')

def finalize(a):
    if (a.out/'plan.json').exists() or (a.out/'approved_links.json').exists():
        raise RuntimeError('Approval is frozen; new --out required')
    dev,_ = dev_rows(a.data)
    meta=json.loads((a.out/'duplicate_candidates.json').read_text(encoding='utf-8'))
    if meta['version']!=AUDIT_VERSION or meta['fingerprint']!=fingerprint(dev):
        raise RuntimeError('Dataset/audit fingerprint mismatch')
    if hashlib.sha256((a.out/'dev_features.npz').read_bytes()).hexdigest()!=meta['feature_sha256']:
        raise RuntimeError('Feature file changed after duplicate audit')
    families=read_csv(a.out/'family_decisions.csv'); pairs=read_csv(a.out/'pair_decisions.csv')
    if set(x['family_id'] for x in families)!=set(meta['families']):
        raise RuntimeError('family_decisions.csv row IDs changed; re-run audit in new --out')
    if set(x['pair_id'] for x in pairs)!=set(meta['pairs']):
        raise RuntimeError('pair_decisions.csv row IDs changed; re-run audit in new --out')
    byrel={r['rel']:i for i,r in enumerate(dev)}
    approved=[]; rejected=[]; pending=[]
    def action(v):
        v=v.strip().lower()
        if v in ('approve','yes','evet','1'):return 'approve'
        if v in ('reject','no','hayir','0'):return 'reject'
        return 'review'
    for f in families:
        ref=meta['families'][f['family_id']]
        if ref['members']!=f['members'] or ref['label']!=f['label']:
            raise RuntimeError(f'Family candidate details changed: {f["family_id"]}')
        decision=action(f['decision'])
        if decision=='review': pending.append('family '+f['family_id']);continue
        members=ref['members'].split('|')
        if decision=='approve':
            approved.extend([[members[0],x] for x in members[1:]])
        else: rejected.append(['family',f['family_id']])
    for p in pairs:
        ref=meta['pairs'][p['pair_id']]
        if ref!=[p['rel_a'],p['rel_b']]:
            raise RuntimeError(f'Pair candidate details changed: {p["pair_id"]}')
        decision=action(p['decision'])
        if decision=='review': pending.append('pair '+p['pair_id']);continue
        if decision=='approve': approved.append(ref)
        else: rejected.append(ref)
    if pending and not a.allow_unreviewed:
        raise RuntimeError(f'{len(pending)} manual review items unresolved. Fill decision=approve/reject, or explicit --allow-unreviewed (weaker protocol). First: {pending[:8]}')
    # Check exact and approvals plus rejected pairs for transitive inconsistencies.
    from devcv import union_find_groups
    seen={}; edges=[]
    for i,r in enumerate(dev):
        h=r['pixel_sha256']
        if h in seen: edges.append((seen[h],i))
        else:seen[h]=i
    edges.extend((byrel[a],byrel[b]) for a,b in approved)
    g=union_find_groups(len(dev),edges)
    conflicts=[x for x in rejected if len(x)==2 and x[0] in byrel and x[1] in byrel and g[byrel[x[0]]]==g[byrel[x[1]]]]
    if conflicts:
        raise RuntimeError(f'Approved/rejected links conflict transitively; resolve review decisions. First {conflicts[:3]}')
    links={'version':AUDIT_VERSION,'fingerprint':fingerprint(dev),'candidate_sha256':sha(meta),
           'feature_sha256':meta['feature_sha256'],'approved_edges':approved,
           'rejected':rejected,'pending':pending,'n_exact_edges':len(edges)-len(approved),
           'n_groups':len(set(g.tolist())),'largest_group':int(max(Counter(g.tolist()).values())),
           'note':'Manual variant-family judgments; no patient IDs; decisions do not certify completeness.'}
    links['signature']=sha(links)
    save_json(a.out/'approved_links.json',links)
    print('FINALIZED:',json.dumps({k:v for k,v in links.items() if k not in ['approved_edges','rejected']},ensure_ascii=False))

def mark_known(a):
    """Apply ONLY source families already visually provided and verified in the conversation.
    Explicit user invocation; does not settle any other candidate.
    """
    if (a.out/'plan.json').exists() or (a.out/'approved_links.json').exists():
        raise RuntimeError('Review is frozen')
    dev,_=dev_rows(a.data)
    m=json.loads((a.out/'duplicate_candidates.json').read_text(encoding='utf-8'))
    if m['fingerprint']!=fingerprint(dev):
        raise RuntimeError('Dataset fingerprint changed')
    families=read_csv(a.out/'family_decisions.csv')
    pairs=read_csv(a.out/'pair_decisions.csv')
    known={'2','13','16','17','19'}
    approved=0
    approved_rel_groups=[]
    for f in families:
        if f['label']=='normal' and f['copy_root'] in known:
            f['decision']='approve';approved+=1
            approved_rel_groups.append(set(f['members'].split('|')))
    for p in pairs:
        if any(p['rel_a'] in group and p['rel_b'] in group for group in approved_rel_groups):
            p['decision']='approve'
    write_csv(a.out/'family_decisions.csv',list(families[0]) if families else ['family_id','label','copy_root','n','spans_orig_train_valid','members','decision'],families)
    write_csv(a.out/'pair_decisions.csv',list(pairs[0]) if pairs else ['pair_id','rel_a','rel_b','label_a','label_b','hamming','imagenet_cosine','pixel_corr96','reason','decision'],pairs)
    print('KNOWN VISUALLY REVIEWED NORMAL FAMILIES MARKED:',approved)
    print('All other candidate rows still require explicit review. Do not auto-approve by filename alone.')

def legacy(a):
    """Count visually approved groups that were split by the OLD CV plan.
    Diagnostic only; does not turn old OOF scores into unbiased performance estimates.
    """
    if a.legacy_predictions is None:
        raise RuntimeError('--legacy-predictions required for legacy stage')
    dev,_=dev_rows(a.data)
    links=json.loads((a.out/'approved_links.json').read_text(encoding='utf-8'))
    if links['fingerprint']!=fingerprint(dev):raise RuntimeError('Dataset changed since approvals')
    from devcv import union_find_groups
    byrel={r['rel']:i for i,r in enumerate(dev)}
    pixel={}; edges=[]
    for i,r in enumerate(dev):
        h=r['pixel_sha256']
        if h in pixel:edges.append((pixel[h],i))
        else:pixel[h]=i
    edges.extend((byrel[x],byrel[y]) for x,y in links['approved_edges'])
    gs=union_find_groups(len(dev),edges)
    old=read_csv(a.legacy_predictions)
    mapping={r['relative_path'].replace('\\','/'):r['fold'] for r in old}
    if len(mapping)<len(dev):
        raise RuntimeError('Legacy CSV does not cover all development images')
    record=[]
    for gi in sorted(set(gs.tolist())):
        idxs=np.flatnonzero(gs==gi)
        if len(idxs)<2:continue
        folds={mapping[dev[i]['rel']] for i in idxs}
        record.append({'group':int(gi),'images':int(len(idxs)),
                       'old_folds':'|'.join(sorted(folds)),
                       'old_fold_leakage':int(len(folds)>1),
                       'rel_paths':'|'.join(dev[i]['rel'] for i in idxs)})
    write_csv(a.out/'legacy_crossfold_reviewed_groups.csv',
              ['group','images','old_folds','old_fold_leakage','rel_paths'],record)
    summary={'reviewed_groups_ge2':len(record),
             'legacy_cross_fold_groups':sum(r['old_fold_leakage'] for r in record),
             'images_in_cross_fold_reviewed_groups':sum(r['images'] for r in record if r['old_fold_leakage']),
             'note':'Cross-fold duplicate-family contamination check; not patient-disjoint and not a corrected accuracy.'}
    save_json(a.out/'legacy_crossfold_summary.json',summary)
    print('LEGACY COPY-FAMILY LEAKAGE:',json.dumps(summary,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('audit','mark-known','finalize','legacy'),required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-cosine',type=float,default=.985)
    p.add_argument('--max-hamming',type=int,default=3)
    p.add_argument('--sheet-limit',type=int,default=120)
    p.add_argument('--legacy-predictions',type=Path,default=None)
    p.add_argument('--allow-unreviewed',action='store_true',help='Not recommended; records unreviewed risk')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    {'audit':audit,'mark-known':mark_known,'finalize':finalize,'legacy':legacy}[a.stage](a)
if __name__=='__main__': main()
