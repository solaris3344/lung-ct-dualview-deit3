#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gelistirme kumesi (train+valid = 558 goruntu) uzerinde benzerlik gruplu, eslenik CV altyapisi.

TEST bolumu bu betikte HIC okunmaz (yalnizca 'testpairs' asamasi gorsel denetim icin
test-train dHash yakin ciftlerinin resmini cizer; model, esik veya secim icin kullanilmaz).

Asamalar (sirayla):
  features : Ince ayarsiz ImageNet DeiT3 oznitelikleri (gruplama icin, etiket gormemis uzay)
  group    : Esik taramasi + kontak sayfalari; --sim-threshold verilirse groups.csv yazilir
  testpairs: (opsiyonel) Test-train dHash<=3 ciftlerinin kontak sayfasi, yalnizca gorsel denetim
  plan     : StratifiedGroupKFold kat plani (plan.json), bir kez dondurulur
  fit      : Bir kol (cozunurluk x en-boy) icin her tohum ve katta ImageNet agirliklarindan egitim,
             SABIT epok (erken durdurma/secim yok), OOF olasiliklari kaydedilir; kaldigi yerden devam eder
  compare  : Kollarin eslenik karsilastirmasi: tohum ortalamasi, McNemar, grup-kume bootstrap GA,
             egitim-katina benzerlik ceyreklerine gore dogruluk
  smoke    : Sentetik veriyle uctan uca kod testi (CPU)

Not: Gruplar hasta kimligi DEGILDIR. Bu protokol hasta-ayrik oldugu iddia edilerek sunulmamalidir.
"""
from __future__ import annotations
import torch  # Windows'ta bilimsel modullerden once
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse, csv, hashlib, json, math, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABELS = ('adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma')
SHORT = ('ADC', 'LCC', 'NOR', 'SCC')
VERSION = 'CHEST_DUPLICATE_AWARE_GROUP_CV_V3'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FEAT_MODEL = 'deit3_base_patch16_224.fb_in22k_ft_in1k'


# ----------------------------------------------------------------------------- yardimcilar
def load_manifest_fn():
    folder = Path(__file__).resolve().parent
    if not (folder / 'audit_dataset.py').is_file():
        raise FileNotFoundError('V3 bundle missing audit_dataset.py')
    sys.path.insert(0, str(folder))
    from audit_dataset import read_manifest, CLASSES
    if tuple(CLASSES) != LABELS:
        raise RuntimeError('Sinif sirasi audit_dataset ile uyusmuyor')
    return read_manifest


def dev_rows(data: Path):
    """Sadece train/valid okur. Test klasoru listelenmez veya acilmaz.

    TEST ile piksel bazinda cakisma denetimi bu gelistirme akisi disinda,
    bagimsiz veri denetiminde daha once yapilmis olmalidir.
    """
    load_manifest_fn()  # kardes modulu sys.path'e ekle
    from audit_dataset import canonical_label, dhash, EXTS
    dev = []
    for split in ('train', 'valid'):
        folder = data / split
        if not folder.is_dir():
            raise FileNotFoundError(f'Eksik gelistirme klasoru: {folder}')
        for cls in sorted(folder.iterdir()):
            if not cls.is_dir():
                continue
            label = canonical_label(cls.name)
            for p in sorted(cls.rglob('*')):
                if not p.is_file() or p.suffix.lower() not in EXTS:
                    continue
                with Image.open(p) as raw:
                    im = ImageOps.exif_transpose(raw).convert('RGB')
                    w, h = im.size
                    pixel_hash = hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()
                    dh = f'{dhash(im):016x}'
                dev.append(dict(split=split, label=label, path=str(p.resolve()),
                                width=w, height=h, pixel_sha256=pixel_hash, dhash64=dh))
    dev.sort(key=lambda r: (r['split'], r['label'], r['path']))
    if not dev:
        raise RuntimeError('Gelistirme goruntusu bulunamadi')
    for r in dev:
        r['rel'] = str(Path(r['path']).resolve().relative_to(data.resolve())).replace('\\', '/')
    print(f'DEV ONLY: train={sum(r["split"] == "train" for r in dev)} '
          f'valid={sum(r["split"] == "valid" for r in dev)}; TEST okunmadi.', flush=True)
    return dev, dev


def fingerprint(dev):
    rec = sorted((r['split'], r['label'], r['rel'], r['pixel_sha256']) for r in dev)
    return hashlib.sha256(json.dumps(rec).encode()).hexdigest()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def pick_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA istendi ama yok')
    return torch.device(name)


# ----------------------------------------------------------------------------- on isleme
class Letterbox:
    """En-boy oranini koruyarak uzun kenari res yapar, siyah (CT arka plani) ile kare tamamlar."""
    def __init__(self, res):
        self.res = res

    def __call__(self, im):
        w, h = im.size
        s = self.res / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (self.res, self.res), (0, 0, 0))
        canvas.paste(im, ((self.res - nw) // 2, (self.res - nh) // 2))
        return canvas


def make_transform(res, aspect, train):
    t = [Letterbox(res)] if aspect == 'letterbox' else \
        [transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC)]
    if train:  # train_eval.py 'baseline' tarifiyle ayni artirma
        t.append(transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03), fill=0))
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


class ImgSet(Dataset):
    _cache = {}

    def __init__(self, rows, idx, transform):
        self.rows, self.idx, self.t = rows, list(idx), transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        p = self.rows[j]['path']
        im = ImgSet._cache.get(p)
        if im is None:
            with Image.open(p) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
            ImgSet._cache[p] = im
        return self.t(im), LABELS.index(self.rows[j]['label']), j


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def stage_features(a):
    if (a.out / 'plan.json').exists():
        raise RuntimeError('Plan dondurulmus. Oznitelikleri degistirmek icin yeni --out kullanin.')
    import timm
    dev, _ = dev_rows(a.data)
    device = pick_device(a.device)
    set_seed(42)
    model = timm.create_model(a.feat_model, pretrained=not a.no_pretrained, num_classes=0).to(device).eval()
    dl = DataLoader(ImgSet(dev, range(len(dev)), make_transform(224, 'letterbox', False)),
                    batch_size=a.batch, shuffle=False, num_workers=0)
    feats = []
    for x, _, _ in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            f = model(x.to(device))
        feats.append(F.normalize(f.float(), dim=1).cpu().numpy())
    Fm = np.concatenate(feats).astype(np.float32)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'dev_features.npz', feats=Fm, rel=np.array([r['rel'] for r in dev]),
                        label=np.array([LABELS.index(r['label']) for r in dev]),
                        fingerprint=fingerprint(dev), feat_model=a.feat_model)
    print(f'FEATURES: {Fm.shape} -> {a.out / "dev_features.npz"}', flush=True)


def load_features(a, dev):
    z = np.load(a.out / 'dev_features.npz', allow_pickle=False)
    if str(z['fingerprint']) != fingerprint(dev) or list(z['rel']) != [r['rel'] for r in dev]:
        raise RuntimeError('dev_features.npz bu veriyle uyusmuyor; features asamasini yeniden calistirin.')
    return z['feats'], str(z['feat_model'])


# ----------------------------------------------------------------------------- model preflight
@torch.no_grad()
def stage_preflight(a):
    """Ayni baslangicli DeiT3'u 224/384'te teknik olarak dener; veri ve TEST okumaz."""
    import timm
    device = pick_device(a.device)
    mid = model_id_for(a)
    model = timm.create_model(mid, pretrained=not a.no_pretrained, num_classes=4).to(device).eval()
    if not hasattr(model, 'set_input_size'):
        raise RuntimeError(f'{mid}: set_input_size yok; timm guncelleyin')
    for resolution in (224, 384):
        model.set_input_size(img_size=(resolution, resolution))
        if a.grad_ckpt:
            if not hasattr(model, 'set_grad_checkpointing'):
                raise RuntimeError('Gradient checkpointing desteklenmiyor')
            model.set_grad_checkpointing(True)
        # Yalnizca geometri/donanimi kontrol eder; egitim belleğini garanti etmez.
        x = torch.zeros((1, 3, resolution, resolution), device=device)
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
            y = model(x)
        if tuple(y.shape) != (1, 4):
            raise RuntimeError(f'Beklenmeyen cikti: {tuple(y.shape)}')
        print(f'PREFLIGHT PASS resolution={resolution} output={tuple(y.shape)} img_size={getattr(getattr(model, "patch_embed", None), "img_size", "unknown")}', flush=True)
        del x, y
        if device.type=='cuda':
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- group
def hamming_matrix(hashes):
    h = np.array([int(x, 16) for x in hashes], dtype=np.uint64)
    x = h[:, None] ^ h[None, :]
    out = np.zeros(x.shape, dtype=np.int16)
    for _ in range(64):
        out += (x & np.uint64(1)).astype(np.int16)
        x = x >> np.uint64(1)
    return out


def union_find_groups(n, edges):
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in edges:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri
    roots = [root(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


def make_groups(S, H, labels, sim_thr, ham_thr, linkage, pixel_hashes=None):
    """Ortalama (veya tam) bagli hiyerarsik kumeleme + dHash/kopya kenarlariyla birlesim."""
    from scipy.cluster.hierarchy import linkage as hlink, fcluster
    from scipy.spatial.distance import squareform
    n = len(S)
    D = np.clip(1.0 - S, 0, 2).astype(np.float64)
    np.fill_diagonal(D, 0)
    Z = hlink(squareform(D, checks=False), method=linkage)
    cl = fcluster(Z, t=1.0 - sim_thr, criterion='distance') - 1
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if cl[i] == cl[j]]
    if ham_thr >= 0:
        ii, jj = np.where(np.triu(H <= ham_thr, 1))
        edges += list(zip(ii.tolist(), jj.tolist()))
    else:
        ii, jj = np.array([], dtype=int), np.array([], dtype=int)
    # Piksel ozdesligi grup ayarindan BAGIMSIZ olarak daima korunur.
    exact_edges = 0
    if pixel_hashes is not None:
        seen = {}
        for i, key in enumerate(pixel_hashes):
            if key in seen:
                edges.append((seen[key], i)); exact_edges += 1
            else:
                seen[key] = i
    g = union_find_groups(n, edges)
    cnt = Counter(g.tolist())
    labs = defaultdict(set)
    for i, gi in enumerate(g):
        labs[gi].add(int(labels[i]))
    S2 = S.copy(); np.fill_diagonal(S2, -2)
    nn = S2.argmax(1)
    return g, dict(sim_threshold=sim_thr, ham_threshold=ham_thr, linkage=linkage,
                   exact_edges=exact_edges, dhash_candidate_edges=len(ii),
                   n_groups=len(cnt), largest=max(cnt.values()),
                   n_groups_ge2=sum(v >= 2 for v in cnt.values()),
                   images_in_groups_ge2=sum(v for v in cnt.values() if v >= 2),
                   multilabel_groups=sum(len(v) > 1 for v in labs.values()),
                   images_in_multilabel_groups=sum(cnt[k] for k, v in labs.items() if len(v) > 1),
                   nn_in_same_group=float(np.mean(g[nn] == g)))


def contact_sheet(rows, pairs, path, title, thumb=180):
    if not pairs:
        return
    pad, cap = 6, 30
    W = 2 * thumb + 3 * pad
    Hh = len(pairs) * (thumb + cap + pad) + pad + 24
    sheet = Image.new('RGB', (W, Hh), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), title, fill=(0, 0, 0))
    y = 24
    for (i, j, txt) in pairs:
        for k, idx in enumerate((i, j)):
            r = rows[idx]
            with Image.open(r['path']) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
            im = Letterbox(thumb)(im)
            sheet.paste(im, (pad + k * (thumb + pad), y))
            d.text((pad + k * (thumb + pad), y + thumb + 2),
                   f"{SHORT[LABELS.index(r['label'])]} {r.get('split', '')} {Path(r['path']).name[:22]}",
                   fill=(0, 0, 0))
        d.text((pad, y + thumb + 15), txt, fill=(180, 0, 0))
        y += thumb + cap + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def stage_group(a):
    """Pixel-identical + explicitly reviewed copy/crop/annotation links ONLY.

    dHash and ImageNet are candidate detectors, never automatic patient links.
    """
    if (a.out / 'plan.json').exists():
        raise RuntimeError('Plan frozen: new --out needed for regrouping')
    if a.sim_threshold is not None or a.max_hamming >= 0:
        raise RuntimeError('V3 automatic cosine/dHash grouping DISABLED. Run duplicate_review.py audit/finalize first; use --max-hamming -1 (default).')
    dev,_=dev_rows(a.data)
    Fm,feat_model=load_features(a,dev)
    links_path=a.out/'approved_links.json'
    links=json.loads(links_path.read_text(encoding='utf-8'))
    from duplicate_review import sha, AUDIT_VERSION
    if links['version']!=AUDIT_VERSION or links['fingerprint']!=fingerprint(dev):
        raise RuntimeError('Approved links invalid for this dataset')
    if links.get('signature')!=sha({k:v for k,v in links.items() if k!='signature'}):
        raise RuntimeError('Approved links signature mismatch')
    if links['feature_sha256']!=file_digest(a.out/'dev_features.npz'):
        raise RuntimeError('ImageNet features changed since manual review')
    # When audit was finalized with explicit --allow-unreviewed, report stays marked incomplete.
    relmap={r['rel']:i for i,r in enumerate(dev)}
    seen={}; edges=[]
    for i,r in enumerate(dev):
        pix=r['pixel_sha256']
        if pix in seen:edges.append((seen[pix],i))
        else:seen[pix]=i
    exact_edge_n=len(edges)
    for r1,r2 in links['approved_edges']:
        edges.append((relmap[r1],relmap[r2]))
    g=union_find_groups(len(dev),edges)
    cnt=Counter(g.tolist())
    labs=defaultdict(set)
    for i,gi in enumerate(g): labs[gi].add(dev[i]['label'])
    if max(cnt.values())>a.max_group_frac*len(dev):
        raise RuntimeError('Largest group exceeds configured fraction; review required')
    if (a.out/'groups.csv').exists():
        # No silent changes to already prepared groups; require separate run directory.
        raise RuntimeError('groups.csv already exists. Do not overwrite; use new --out')
    with (a.out/'groups.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['rel','split','label','group'])
        for r,gi in zip(dev,g): w.writerow([r['rel'],r['split'],r['label'],int(gi)])
    m={'version':VERSION,'fingerprint':fingerprint(dev),'feat_model':feat_model,
       'reviewed_links_sha256':file_digest(links_path),'approval_signature':links['signature'],
       'n_groups':len(cnt),'largest':max(cnt.values()),
       'n_groups_ge2':sum(v>=2 for v in cnt.values()),
       'images_in_groups_ge2':sum(v for v in cnt.values() if v>=2),
       'mixed_label_groups':sum(len(x)>1 for x in labs.values()),
       'images_in_mixed_label_groups':sum(cnt[k] for k,v in labs.items() if len(v)>1),
       'exact_pixel_edges':exact_edge_n,'manually_approved_edges':len(links['approved_edges']),
       'unreviewed_candidates':len(links['pending']),
       'warning':'Groups are source-image proxies, never verified patients.'}
    save_json(a.out/'groups_meta.json',m)
    print('V3 REVIEWED GROUPS:',json.dumps(m,ensure_ascii=False),flush=True)
    if links['pending']:
        print('WARNING: Unreviewed candidate duplicates remain; generalization may be optimistic.',flush=True)


def stage_testpairs(a):
    """Yalnizca gorsel veri denetimi: test goruntusu -> dHash<=esik egitim komsusu. Model yok, secim yok."""
    # Test okuyan TEK asama budur; kesinlikle bir protokol secim girdisi degildir.
    read_manifest = load_manifest_fn()
    rows = read_manifest(a.data)
    tr = [r for r in rows if r['split'] in ('train', 'valid')]
    te = [r for r in rows if r['split'] == 'test']
    all_rows = tr + te
    htr = np.array([int(r['dhash64'], 16) for r in tr], dtype=np.uint64)
    pairs = []
    for k, r in enumerate(te):
        x = htr ^ np.uint64(int(r['dhash64'], 16))
        d = np.array([bin(int(v)).count('1') for v in x])
        j = int(d.argmin())
        if d[j] <= a.max_hamming:
            pairs.append((len(tr) + k, j, f'dHash={int(d[j])} test->gelistirme'))
    contact_sheet(all_rows, pairs, a.out / 'sheets' / 'test_dev_dhash_pairs.png',
                  'GORSEL DENETIM: test ile dHash yakin gelistirme ciftleri (secim icin kullanmayin)')
    print(f'{len(pairs)} cift cizildi -> {a.out / "sheets" / "test_dev_dhash_pairs.png"}', flush=True)


# ----------------------------------------------------------------------------- plan
def read_groups(a, dev):
    meta = json.loads((a.out / 'groups_meta.json').read_text(encoding='utf-8'))
    if meta['fingerprint'] != fingerprint(dev):
        raise RuntimeError('groups_meta.json bu veriyle uyusmuyor')
    if meta.get('reviewed_links_sha256') != file_digest(a.out / 'approved_links.json'):
        raise RuntimeError('approved_links.json has changed after reviewed grouping')
    g = {}
    with (a.out / 'groups.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            g[row['rel']] = int(row['group'])
    garray=np.array([g[r['rel']] for r in dev])
    if len(g)!=len(dev):
        raise RuntimeError('groups.csv row count changed or duplicates found')
    links=json.loads((a.out/'approved_links.json').read_text(encoding='utf-8'))
    seen={}; e=[]; relmap={r['rel']:i for i,r in enumerate(dev)}
    for i,r in enumerate(dev):
        h=r['pixel_sha256']
        if h in seen:e.append((seen[h],i))
        else:seen[h]=i
    for ra,rb in links['approved_edges']:
        e.append((relmap[ra],relmap[rb]))
    expected=union_find_groups(len(dev),e)
    # Equivalent partition, regardless of numeric group ID choices.
    expected_parts=sorted(tuple(np.where(expected==k)[0].tolist()) for k in np.unique(expected))
    actual_parts=sorted(tuple(np.where(garray==k)[0].tolist()) for k in np.unique(garray))
    if expected_parts!=actual_parts:
        raise RuntimeError('groups.csv does not match EXACT + human-approved links')
    return garray, meta


def file_digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def group_digest(dev, groups):
    if len(groups) != len(dev):
        raise RuntimeError('Grup sayisi veri sayisiyla uyusmuyor')
    rec = [(r['rel'], int(g)) for r, g in zip(dev, groups)]
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()


def plan_digest(plan):
    obj = {k: v for k, v in plan.items() if k != 'signature'}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def stage_plan(a):
    from sklearn.model_selection import StratifiedGroupKFold
    dev, _ = dev_rows(a.data)
    if (a.out / 'plan.json').exists():
        raise RuntimeError('plan.json zaten var ve dondurulmus. Yeni plan icin yeni --out kullanin.')
    groups, meta = read_groups(a, dev)
    y = np.array([LABELS.index(r['label']) for r in dev])
    sg = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.plan_seed)
    folds = []
    for k, (tr, ho) in enumerate(sg.split(np.zeros(len(y)), y, groups)):
        if set(groups[tr]) & set(groups[ho]):
            raise RuntimeError('Grup sizintisi')
        ctr, cho = Counter(y[tr].tolist()), Counter(y[ho].tolist())
        if any(cho[c] < 2 or ctr[c] < 5 for c in range(4)):
            raise RuntimeError(f'Kat {k} sinif kapsamasi yetersiz: train={dict(ctr)} held={dict(cho)}; baska --plan-seed deneyin')
        folds.append(sorted(ho.tolist()))
        print(f'KAT {k}: egitim={len(tr)} {dict(sorted(ctr.items()))} | ayrilan={len(ho)} {dict(sorted(cho.items()))}', flush=True)
    if sorted(i for f in folds for i in f) != list(range(len(dev))):
        raise RuntimeError('Kapsama hatasi')
    plan = dict(version=VERSION, fingerprint=fingerprint(dev), folds=a.folds, plan_seed=a.plan_seed,
                group_meta=meta, group_digest=group_digest(dev, groups),
                features_sha256=file_digest(a.out / 'dev_features.npz'),
                heldout_rel=[[dev[i]['rel'] for i in f] for f in folds])
    plan['signature'] = plan_digest(plan)
    save_json(a.out / 'plan.json', plan)
    print('plan.json donduruldu. Imza:', plan['signature'][:16], flush=True)


def load_plan(a, dev):
    plan = json.loads((a.out / 'plan.json').read_text(encoding='utf-8'))
    if plan['fingerprint'] != fingerprint(dev):
        raise RuntimeError('plan.json bu veriyle uyusmuyor')
    if plan.get('signature') != plan_digest(plan):
        raise RuntimeError('plan.json imzasi tutmuyor; dosya degistirilmis!')
    if plan.get('version') != VERSION:
        raise RuntimeError('Plan surumu farkli; yeni cikti klasorunde yeniden planlayin')
    if plan.get('features_sha256') != file_digest(a.out / 'dev_features.npz'):
        raise RuntimeError('ImageNet ozellik dosyasi dondurulmus plandan sonra degismis!')
    groups, meta = read_groups(a, dev)
    if group_digest(dev, groups) != plan['group_digest'] or meta != plan['group_meta']:
        raise RuntimeError('Gruplama plan dondurulduktan sonra degismis! Yeni --out klasoru gerekli.')
    pos = {r['rel']: i for i, r in enumerate(dev)}
    folds = [np.array([pos[x] for x in f], dtype=np.int64) for f in plan['heldout_rel']]
    held = [int(i) for fold in folds for i in fold]
    if len(held) != len(dev) or sorted(held) != list(range(len(dev))):
        raise RuntimeError('Kat planinda eksik/tekrar goruntu var')
    for ho in folds:
        tr = np.setdiff1d(np.arange(len(dev)), ho)
        if set(groups[ho]) & set(groups[tr]):
            raise RuntimeError('Dondurulmus planda grup sizintisi!')
    return plan, folds


# ----------------------------------------------------------------------------- fit
def model_id_for(a):
    # Cozunurluk kollari ImageNet-224'te AYNI agirliklarla baslar.
    # 384 kolunda set_input_size pozisyon gommesini interpolate eder.
    return a.model_id or FEAT_MODEL


def train_one(a, dev, tr_idx, ho_idx, seed, device, resume_path=None):
    import timm
    set_seed(seed)
    mid = model_id_for(a)
    model = timm.create_model(mid, pretrained=not a.no_pretrained, num_classes=4)
    if a.res != 224:
        if not hasattr(model, 'set_input_size'):
            raise RuntimeError(f'{mid}: set_input_size gerekli; timm surumunu kontrol edin')
        model.set_input_size(img_size=(a.res, a.res))
        print(f'    Pozisyon gommesi ImageNet-224 -> {a.res} olarak uyarlandi', flush=True)
    if a.grad_ckpt:
        if not hasattr(model, 'set_grad_checkpointing'):
            raise RuntimeError(f'{mid}: gradient checkpointing desteklenmiyor')
        model.set_grad_checkpointing(True)
    model = model.to(device)
    head = list(model.get_classifier().parameters())
    hid = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{'params': backbone, 'lr': a.backbone_lr}, {'params': head, 'lr': a.head_lr}],
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-7)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ytr = [LABELS.index(dev[i]['label']) for i in tr_idx]
    cnt = Counter(ytr)
    cw = torch.tensor([math.sqrt(len(ytr) / (4 * cnt[c])) for c in range(4)], dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ImgSet(dev, tr_idx, make_transform(a.res, a.aspect, True)), batch_size=a.batch,
                    shuffle=True, generator=g, num_workers=0, pin_memory=amp, drop_last=False)
    nb = len(dl)
    start_ep = 1
    if resume_path is not None and resume_path.exists():
        # Yalnizca BU betigin URETTIGI yerel dosyayi acin; torch.load pickle
        # genel olarak guvenilmeyen dosyalar icin kullanilmamalidir.
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved.get('seed') != seed or saved.get('epochs') != a.epochs:
            raise RuntimeError('Resume dosyasi bu egitime ait degil. Silip bastan baslayin.')
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        g.set_state(saved['dataloader_rng'])
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        if torch.cuda.is_available() and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        start_ep = int(saved['epoch']) + 1
        del saved
        print(f'    EPOCH RESUME: {start_ep}/{a.epochs}', flush=True)
    for ep in range(start_ep, a.epochs + 1):
        freeze = ep <= a.freeze_epochs
        for p in backbone:
            p.requires_grad_(not freeze)
        model.train(); opt.zero_grad(set_to_none=True)
        tot, n = 0.0, 0
        for step, (x, y, _) in enumerate(dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            win = min(a.accum, nb - (step // a.accum) * a.accum)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = F.cross_entropy(model(x), y, weight=cw)
            if not torch.isfinite(loss):
                raise RuntimeError(f'NaN kayip epok={ep} adim={step}')
            scaler.scale(loss / win).backward()
            tot += float(loss.detach()) * len(y); n += len(y)
            if (step + 1) % a.accum == 0 or step + 1 == nb:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        sched.step()
        if resume_path is not None:
            # Guncel tek bir resume dosyasi; kat tamamlaninca silinir.
            snap = dict(epoch=ep, epochs=a.epochs, seed=seed,
                        model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                        dataloader_rng=g.get_state(), rng_python=random.getstate(),
                        rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            tmp = resume_path.with_suffix('.tmp')
            torch.save(snap, tmp)
            tmp.replace(resume_path)
        print(f'    epok {ep:02d}/{a.epochs} egitimCE={tot / n:.4f}', flush=True)
    # SABIT son epok agirliklari: secim yok, ayrilan kat yalnizca bir kez tahmin edilir
    model.eval()
    dl_ho = DataLoader(ImgSet(dev, ho_idx, make_transform(a.res, a.aspect, False)),
                       batch_size=max(1, a.batch * 2), shuffle=False, num_workers=0)
    probs, idx = [], []
    with torch.no_grad():
        for x, _, j in dl_ho:
            with torch.autocast(device_type=device.type, enabled=amp):
                lo = model(x.to(device))
            probs.append(F.softmax(lo.float(), 1).cpu().numpy()); idx.extend(j.tolist())
    if idx != list(ho_idx):
        raise RuntimeError('Ayrilan kat sirasi bozuldu')
    state = model.state_dict() if a.save_ckpt else None
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(probs), state


def arm_config(a):
    keys = ('res', 'aspect', 'epochs', 'batch', 'accum', 'backbone_lr', 'head_lr', 'weight_decay',
            'freeze_epochs', 'model_id', 'no_pretrained', 'grad_ckpt')
    c = {k: getattr(a, k) for k in keys}
    c['resolved_model'] = model_id_for(a)
    c['weight_source'] = 'shared_ImageNet_224_pretrained_pos_interp' if not a.no_pretrained else 'random_init_smoke'
    return c


def stage_fit(a):
    dev, _ = dev_rows(a.data)
    plan, folds = load_plan(a, dev)
    device = pick_device(a.device)
    arm = a.out / 'arms' / a.arm
    arm.mkdir(parents=True, exist_ok=True)
    cfg = arm_config(a)
    cfg_path = arm / 'arm_config.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding='utf-8'))
        if old['config'] != cfg or old['plan_signature'] != plan['signature']:
            raise RuntimeError('Bu kol farkli ayarlarla baslatilmis. Yeni --arm adi kullanin.')
    else:
        save_json(cfg_path, {'config': cfg, 'plan_signature': plan['signature'], 'version': VERSION})
    n = len(dev)
    for seed in a.seeds:
        for k, ho in enumerate(folds):
            dest = arm / f'seed{seed}_fold{k}.npz'
            if dest.exists():
                print(f'[{a.arm}] tohum={seed} kat={k} mevcut, atlaniyor', flush=True)
                continue
            tr = np.setdiff1d(np.arange(n), ho)
            print(f'[{a.arm}] tohum={seed} kat={k} egitim={len(tr)} ayrilan={len(ho)} model={cfg["resolved_model"]}', flush=True)
            t0 = time.time()
            resume_path = arm / f'seed{seed}_fold{k}.resume.pt'
            probs, state = train_one(a, dev, tr, ho, seed + k * 101, device, resume_path=resume_path)
            # Tahminler once gecici dosyaya yazilir: kesinti corrupt OOF birakmaz.
            temp_out = arm / f'seed{seed}_fold{k}.staging.npz'
            np.savez_compressed(temp_out, probs=probs, idx=ho, seed=seed, fold=k, plan_signature=plan['signature'],
                                seconds=time.time() - t0)
            temp_out.replace(dest)
            resume_path.unlink(missing_ok=True)
            if state is not None:
                torch.save(state, arm / f'seed{seed}_fold{k}.pt')
            y = np.array([LABELS.index(dev[i]['label']) for i in ho])
            print(f'[{a.arm}] tohum={seed} kat={k} katDogruluk={np.mean(probs.argmax(1) == y) * 100:.2f}% '
                  f'sure={time.time() - t0:.0f}s', flush=True)
    print('FIT tamam (yalnizca gelistirme verisi).', flush=True)


# ----------------------------------------------------------------------------- compare
def metrics(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    out = dict(acc=float(accuracy_score(y, pred)), bal_acc=float(balanced_accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
               recall={SHORT[c]: float(cm[c, c] / max(1, cm[c].sum())) for c in range(4)},
               confusion=cm.tolist())
    try:
        out['macro_auc'] = float(roc_auc_score(y, p, multi_class='ovr', average='macro', labels=[0, 1, 2, 3]))
    except ValueError:
        out['macro_auc'] = None
    return out


def load_arm(a, dev, plan, name):
    arm = a.out / 'arms' / name
    cfg = json.loads((arm / 'arm_config.json').read_text(encoding='utf-8'))
    if cfg['plan_signature'] != plan['signature']:
        raise RuntimeError(f'{name}: plan imzasi uyusmuyor')
    per_seed = {}
    pos = {r['rel']: i for i, r in enumerate(dev)}
    held = [np.array([pos[x] for x in fold], dtype=np.int64) for fold in plan['heldout_rel']]
    observed = set()
    for f in sorted(arm.glob('seed*_fold*.npz')):
        with np.load(f, allow_pickle=False) as z:
            if str(z['plan_signature']) != plan['signature']:
                raise RuntimeError(f'{f}: plan imzasi uyusmuyor')
            s, fold = int(z['seed']), int(z['fold'])
            idx, prob = np.asarray(z['idx'], dtype=np.int64), np.asarray(z['probs'], dtype=np.float64)
            if fold < 0 or fold >= len(held) or not np.array_equal(idx, held[fold]):
                raise RuntimeError(f'{f}: ayrilan fold indeksleri uyusmuyor')
            if prob.shape != (len(idx), 4) or not np.isfinite(prob).all():
                raise RuntimeError(f'{f}: olasiliklar gecersiz')
            if np.max(np.abs(prob.sum(axis=1) - 1)) > 0.001:
                raise RuntimeError(f'{f}: olasiliklar 1 toplamiyor')
            if (s, fold) in observed:
                raise RuntimeError(f'{f}: ayni tohum/kat iki kez yuklenmis')
            observed.add((s, fold))
            P = per_seed.setdefault(s, np.full((len(dev), 4), np.nan))
            P[idx] = prob
    complete = {s: P for s, P in per_seed.items() if not np.isnan(P).any()}
    if not complete:
        raise RuntimeError(f'{name}: tamamlanmis tohum yok')
    return complete, cfg['config']


def stage_compare(a):
    from scipy.stats import binomtest
    dev, _ = dev_rows(a.data)
    plan, folds = load_plan(a, dev)
    groups, _ = read_groups(a, dev)
    y = np.array([LABELS.index(r['label']) for r in dev])
    arms = {}
    for name in a.arms:
        arms[name] = load_arm(a, dev, plan, name)
    common = sorted(set.intersection(*[set(v[0]) for v in arms.values()]))
    if not common:
        raise RuntimeError('Kollar arasinda ortak tamamlanmis tohum yok')
    if set(common) != set(a.seeds):
        raise RuntimeError(f'Karsilastirma icin istenen tohumlar tamamlanmali. istenen={a.seeds}, ortak={common}')
    report = {'common_seeds': common, 'n_dev': len(dev), 'plan_signature': plan['signature'],
              'evidence_level': 'Gelistirme ici gruplu CV; goruntu duzeyi; hasta-ayrik DEGIL', 'arms': {}}
    ens = {}
    for name, (per_seed, cfg) in arms.items():
        ms = [metrics(y, per_seed[s]) for s in common]
        Pm = np.mean([per_seed[s] for s in common], axis=0)
        ens[name] = Pm
        agg = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0)
               for k in ('acc', 'bal_acc', 'macro_f1', 'macro_auc')}
        report['arms'][name] = {'config': cfg, 'per_seed_mean_sd': agg, 'seed_ensemble': metrics(y, Pm)}

    # Ayrilan goruntunun KENDI egitim katina ImageNet benzerligi (protokol altinda genelleme egrisi)
    Fm, _ = load_features(a, dev)
    sim_to_train = np.zeros(len(dev))
    for ho in folds:
        tr = np.setdiff1d(np.arange(len(dev)), ho)
        sim_to_train[ho] = (Fm[ho] @ Fm[tr].T).max(1)
    q = np.quantile(sim_to_train, [0.25, 0.5, 0.75]); b = np.digitize(sim_to_train, q)
    report['strata_sim_to_train_fold'] = {
        name: {qn: {'n': int((b == k).sum()), 'acc': float(np.mean(ens[name][b == k].argmax(1) == y[b == k]))}
               for k, qn in enumerate(('Q1_en_uzak', 'Q2', 'Q3', 'Q4_en_yakin'))} for name in ens}

    names = list(a.arms)
    if len(names) >= 2:
        A, B = names[0], names[1]
        ca, cb = ens[A].argmax(1) == y, ens[B].argmax(1) == y
        b01, b10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        mcn = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        per_seed_diff = [float(np.mean(arms[B][0][s].argmax(1) == y) - np.mean(arms[A][0][s].argmax(1) == y)) for s in common]
        # Grup-kume bootstrap: goruntuler bagimsiz olmadigi icin gruplari yeniden ornekle
        rng = np.random.default_rng(a.boot_seed)
        ug = np.unique(groups)
        members = {gi: np.where(groups == gi)[0] for gi in ug}
        from sklearn.metrics import balanced_accuracy_score, f1_score
        diffs = {'acc': [], 'bal_acc': [], 'macro_f1': []}
        pa, pb = ens[A].argmax(1), ens[B].argmax(1)
        for _ in range(a.boot):
            idx = np.concatenate([members[gi] for gi in rng.choice(ug, size=len(ug), replace=True)])
            yy = y[idx]
            if len(np.unique(yy)) < 4:
                continue
            diffs['acc'].append(np.mean(pb[idx] == yy) - np.mean(pa[idx] == yy))
            diffs['bal_acc'].append(balanced_accuracy_score(yy, pb[idx]) - balanced_accuracy_score(yy, pa[idx]))
            diffs['macro_f1'].append(f1_score(yy, pb[idx], average='macro') - f1_score(yy, pa[idx], average='macro'))
        report['paired'] = {
            'A': A, 'B': B, 'direction': 'B eksi A',
            'mcnemar_exact': {'A_dogru_B_yanlis': b10, 'A_yanlis_B_dogru': b01, 'p': float(mcn)},
            'per_seed_acc_diff': per_seed_diff,
            'group_bootstrap_95ci': {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                     for k, v in diffs.items() if v},
            'bootstrap_valid_resamples': len(diffs['acc'])}
    out = a.out / ('compare_' + '_vs_'.join(names) + '.json')
    save_json(out, report)

    print('\nGELISTIRME ICI GRUPLU CV (goruntu duzeyi, hasta-ayrik degil). Tohumlar:', common, flush=True)
    for name, r in report['arms'].items():
        m, e = r['per_seed_mean_sd'], r['seed_ensemble']
        print(f'{name:18s} acc={m["acc"][0] * 100:.2f}+-{m["acc"][1] * 100:.2f}  balAcc={m["bal_acc"][0] * 100:.2f}  '
              f'F1={m["macro_f1"][0]:.4f}  AUC={m["macro_auc"][0]:.4f} | topluluk acc={e["acc"] * 100:.2f} '
              f'recall={e["recall"]}', flush=True)
        st = report['strata_sim_to_train_fold'][name]
        print('   benzerlik ceyrekleri:', {k: f'{v["acc"] * 100:.1f}% (n={v["n"]})' for k, v in st.items()}, flush=True)
    if 'paired' in report:
        p = report['paired']
        print(f'ESLENIK {p["B"]} - {p["A"]}: McNemar {p["mcnemar_exact"]}  tohum farklari={["%.4f" % d for d in p["per_seed_acc_diff"]]}', flush=True)
        print('   grup bootstrap %95 GA:', p['group_bootstrap_95ci'], flush=True)
    print('Kaydedildi:', out, flush=True)


# ----------------------------------------------------------------------------- smoke
def stage_smoke(a):
    from smoke_v3 import run_smoke
    run_smoke()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('features', 'group', 'testpairs', 'plan', 'fit', 'compare', 'preflight', 'smoke'))
    ap.add_argument('--data', type=Path, default=Path('datasets/chest_ct/Data_exact_clean_v1_1'))
    ap.add_argument('--out', type=Path, default=Path('runs/chest_dev_cv_v1'))
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--batch', type=int, default=8)
    # features / group
    ap.add_argument('--feat-model', default=FEAT_MODEL)
    ap.add_argument('--sweep', type=float, nargs='+', default=[0.97, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83])
    ap.add_argument('--sim-threshold', type=float, default=None)
    ap.add_argument('--max-hamming', type=int, default=-1)
    ap.add_argument('--linkage', default='average', choices=('average', 'complete', 'single'))
    ap.add_argument('--sheet-n', type=int, default=30)
    ap.add_argument('--max-group-frac', type=float, default=0.15)
    # plan
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--plan-seed', type=int, default=2026)
    # fit
    ap.add_argument('--arm', default=None)
    ap.add_argument('--res', type=int, default=224, choices=(224, 384))
    ap.add_argument('--aspect', default='stretch', choices=('stretch', 'letterbox'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--backbone-lr', type=float, default=3e-5)
    ap.add_argument('--head-lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.02)
    ap.add_argument('--freeze-epochs', type=int, default=1)
    ap.add_argument('--grad-ckpt', action='store_true')
    ap.add_argument('--save-ckpt', action='store_true')
    ap.add_argument('--model-id', default=None, help='Yalnizca smoke/test icin; normalde deit3_base_patch16_{res} kullanilir')
    ap.add_argument('--no-pretrained', action='store_true', help='Yalnizca smoke icin')
    # compare
    ap.add_argument('--arms', nargs='+', default=None)
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--boot-seed', type=int, default=7)
    ap.add_argument('--keep-smoke', action='store_true')
    a = ap.parse_args()
    if a.stage != 'smoke':
        a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'fit' and not a.arm:
        raise ValueError('--arm gerekli (or. d224_stretch)')
    if a.stage == 'compare' and not a.arms:
        raise ValueError('--arms gerekli')
    {'features': stage_features, 'group': stage_group, 'testpairs': stage_testpairs, 'plan': stage_plan,
     'fit': stage_fit, 'compare': stage_compare, 'preflight': stage_preflight, 'smoke': stage_smoke}[a.stage](a)


if __name__ == '__main__':
    main()
