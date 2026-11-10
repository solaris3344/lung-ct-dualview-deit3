#!/usr/bin/env python3
"""Audit the 4-class Kaggle Chest CT dataset (not a patient-level leakage audit).

Usage: python audit_dataset.py --data datasets/chest_ct/Data --out runs/chest_ct_audit
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

SPLITS = ('train','valid','test')
CLASSES = ('adenocarcinoma','large.cell.carcinoma','normal','squamous.cell.carcinoma')
EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def canonical_label(directory: str) -> str:
    s = directory.strip().lower().replace('_','.').replace(' ','')
    if s.startswith('adenocarcinoma'):
        return 'adenocarcinoma'
    if s.startswith('large.cell.carcinoma') or s.startswith('largecellcarcinoma'):
        return 'large.cell.carcinoma'
    if s == 'normal':
        return 'normal'
    if s.startswith('squamous.cell.carcinoma') or s.startswith('squamouscellcarcinoma'):
        return 'squamous.cell.carcinoma'
    raise ValueError(f'UNRECOGNISED class folder: {directory!r}; do not silently relabel')

def dhash(im: Image.Image) -> int:
    arr = np.asarray(ImageOps.grayscale(im).resize((9,8),Image.Resampling.LANCZOS), dtype=np.int16)
    return sum(int(v)<<k for k,v in enumerate((arr[:,:-1] > arr[:,1:]).reshape(-1)))

def read_manifest(data_dir: Path) -> list[dict]:
    rows=[]
    for split in SPLITS:
        base=data_dir/split
        if not base.is_dir(): raise FileNotFoundError(f'Missing split directory: {base}')
        for folder in sorted(base.iterdir()):
            if not folder.is_dir(): continue
            label=canonical_label(folder.name)
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or p.suffix.lower() not in EXTS: continue
                with Image.open(p) as raw:
                    im=ImageOps.exif_transpose(raw).convert('RGB')
                    size=im.size
                    phash=hashlib.sha256(im.tobytes()+str(size).encode()).hexdigest()
                    perceptual=dhash(im)
                byhash=hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append(dict(split=split,label=label,folder=folder.name,path=str(p.resolve()),
                                 width=size[0],height=size[1],file_sha256=byhash,
                                 pixel_sha256=phash,dhash64=f'{perceptual:016x}'))
    return rows

def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

def duplicates(rows, field):
    groups=defaultdict(list)
    for x in rows: groups[x[field]].append(x)
    result=[]
    for h, group in groups.items():
        if len({g['split'] for g in group})<2: continue
        for g in group:
            result.append(dict(hash_type=field,hash=h,**{k:g[k] for k in ('split','label','path')}))
    return result

def audit(data_dir:Path,out:Path, near_threshold:int=3):
    rows=read_manifest(data_dir)
    count=Counter((r['split'],r['label']) for r in rows)
    print('\nCLASS COUNTS')
    for s in SPLITS:
        print(s, ' '.join(f'{c}={count[(s,c)]}' for c in CLASSES),f'TOTAL={sum(count[(s,c)] for c in CLASSES)}')
    print('ALL:', len(rows))
    if not rows: raise RuntimeError('No images found')
    for s in SPLITS:
        missing=[c for c in CLASSES if count[s,c]==0]
        if missing: raise RuntimeError(f'Missing classes in {s}: {missing}')
    file_dup=duplicates(rows,'file_sha256'); pixel_dup=duplicates(rows,'pixel_sha256')
    exact_keys={(r['split'],r['path']) for r in file_dup+pixel_dup}
    print('CROSS SPLIT EXACT duplicate files:',len({r['hash'] for r in file_dup}))
    print('CROSS SPLIT IDENTICAL pixels:',len({r['hash'] for r in pixel_dup}))
    print('CROSS SPLIT paths involved:',len(exact_keys))
    # Cross-split perceptual flags; Hamming <= 3 is a conservative lead, not leakage proof.
    near=[]
    sorted_rows=sorted(rows,key=lambda r:(r['split'],r['label'],r['path']))
    for i,a in enumerate(sorted_rows):
        ha=int(a['dhash64'],16)
        for b in sorted_rows[i+1:]:
            if a['split']==b['split']: continue
            d=(ha^int(b['dhash64'],16)).bit_count()
            if d<=near_threshold and a['pixel_sha256']!=b['pixel_sha256']:
                near.append(dict(distance=d,split_a=a['split'],class_a=a['label'],path_a=a['path'],
                                 split_b=b['split'],class_b=b['label'],path_b=b['path']))
    near.sort(key=lambda r:(r['distance'],r['path_a'],r['path_b']))
    print(f'CROSS SPLIT dHash Hamming <= {near_threshold} CANDIDATE PAIRS:',len(near))
    print('WARNING: perceptual flags can be adjacent slices or common backgrounds; review manually.')
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'manifest.csv',rows,['split','label','folder','path','width','height','file_sha256','pixel_sha256','dhash64'])
    save_csv(out/'cross_split_file_duplicates.csv',file_dup,['hash_type','hash','split','label','path'])
    save_csv(out/'cross_split_pixel_duplicates.csv',pixel_dup,['hash_type','hash','split','label','path'])
    save_csv(out/'near_duplicate_candidates.csv',near,['distance','split_a','class_a','path_a','split_b','class_b','path_b'])
    summary=dict(total_images=len(rows),counts={s:{c:count[(s,c)] for c in CLASSES} for s in SPLITS},
                 cross_split_exact_file_groups=len(set(r['hash'] for r in file_dup)),
                 cross_split_identical_pixel_groups=len(set(r['hash'] for r in pixel_dup)),
                 cross_split_exact_paths=len(exact_keys),
                 near_dhash_candidate_pairs=len(near),near_dhash_max_distance=near_threshold,
                 caveat='No patient IDs. The audit detects identical files/pixels and flags some near-duplicates; cannot certify patient disjointness.')
    (out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if exact_keys:
        print('\nWARNING: EXACT CROSS-SPLIT DUPLICATES PRESENT — DO NOT START TRAINING')
    else:
        print('\nNo exact cross-split duplicates detected; patient-level independence remains UNVERIFIED.')
    print('Audit saved to:',out.resolve())
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--out',default=Path('runs/chest_ct_audit'),type=Path)
    p.add_argument('--near-hamming',default=3,type=int)
    a=p.parse_args()
    audit(a.data,a.out,a.near_hamming)
if __name__=='__main__':main()
