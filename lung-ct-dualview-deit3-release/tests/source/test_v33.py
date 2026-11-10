"""Lightweight CPU tests: no dataset, GPU or pretrained downloads required."""
import csv, json, tempfile, types, unittest
from pathlib import Path
import numpy as np
from experiment_v33 import (SIGNATURE, VIEW_NAMES, select_views, validate_prob,
                            compare, calc_metric, reconstruct_baseline, load_baseline_csv, load_data)

class V33Tests(unittest.TestCase):
    def test_selector_inner_oof_and_cache(self):
        rng=np.random.default_rng(37);n=90;d=12
        y=np.arange(n)%4;groups=np.array([str(i//2) for i in range(n)])
        X=rng.normal(size=(n,7,d)).astype(np.float32)
        folds=[np.arange(n)[np.arange(n)//2%5==k] for k in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            o=Path(directory)
            for k in range(5):
                c=select_views(X,y,groups,folds,k,o,'abc')
                self.assertEqual(c.shape,(n,))
                self.assertTrue(((c>=1)&(c<=6)).all())
                np.testing.assert_array_equal(c,select_views(X,y,groups,folds,k,o,'abc'))

    def test_prob_and_bootstrap(self):
        rng=np.random.default_rng(4);y=np.arange(40)%4
        B=np.eye(4)[y]*.8+.05
        P=B.copy();P[0]=np.array([.05,.85,.05,.05]);P[4]=np.array([.9,.05,.025,.025])
        self.assertEqual(calc_metric(y,B)['correct'],40)
        self.assertEqual(calc_metric(y,P)['correct'],39)
        comparison=compare(y,np.arange(40)//2,B,P,bootstrap=60)
        self.assertEqual(comparison['lost'],1)
        self.assertEqual(comparison['rescued'],0)
        with self.assertRaises(ValueError):validate_prob(np.array([[.4,.5,.4,.5]]))

    def test_full_manifest_and_frozen_plan_loading(self):
        rng=np.random.default_rng(5)
        n=558
        rel=[f'train/{["adenocarcinoma","large.cell.carcinoma","normal","squamous.cell.carcinoma"][i%4]}/image{i:03}.png' for i in range(n)]
        groups=np.array([str(i//2) if i<100 else str(i-50) for i in range(n)])
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            csv_file=d/'meta.csv'
            with csv_file.open('w',newline='',encoding='utf-8') as h:
                w=csv.writer(h);w.writerow(['relative_path','class_folder'])
                for r in rel:w.writerow([r,r.split('/')[1]])
            groupfile=d/'groups.csv'
            with groupfile.open('w',newline='',encoding='utf-8') as h:
                w=csv.writer(h);w.writerow(['rel','group'])
                for r,g in zip(rel,groups):w.writerow([r,g])
            helds=[]
            for k in range(5):
                helds.append([rel[i] for i in range(n) if int(groups[i])%5==k])
            plan=d/'plan.json';plan.write_text(json.dumps({'signature':SIGNATURE,'heldout_rel':helds}))
            feat=d/'feat.npz'
            vals=rng.standard_normal((n,7,768)).astype('float32')
            np.savez_compressed(feat,signature=SIGNATURE,rel=np.array(rel),views=np.array(VIEW_NAMES),features=vals)
            a=types.SimpleNamespace(csv=csv_file,groups=groupfile,plan=plan,features=feat)
            rr,yy,gg,ff,XX,fp=load_data(a)
            self.assertEqual(len(rr),558)
            self.assertEqual(XX.shape,(558,7,768))
            self.assertEqual(len(set(gg)),508)
            self.assertEqual(len(ff),5)

    def test_recover_rejects_nonmatching_plan(self):
        y=np.arange(558)%4;rel=[f'train/fake_{i}.png' for i in range(558)]
        folds=[np.arange(558)[np.arange(558)%5==k] for k in range(5)]
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'old';root.mkdir();out=base/'out';out.mkdir()
            for seed in (42,52,62):
                for k,idx in enumerate(folds):
                    p=np.eye(4)[y[idx]]*.8+.05
                    np.savez_compressed(root/f'seed{seed}_fold{k}.npz',indices=idx,probs=p,
                                        plan_signature='WRONG',seed=seed,fold=k)
            a=types.SimpleNamespace(baseline_oof=None,out=out,baseline_root=root)
            B,where=reconstruct_baseline(a,rel,y,folds)
            self.assertIsNone(B)
            self.assertIn('0 uniquely',where)

if __name__=='__main__': unittest.main(verbosity=2)
