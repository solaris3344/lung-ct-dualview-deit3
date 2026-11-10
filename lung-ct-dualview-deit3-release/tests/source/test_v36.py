# -*- coding: utf-8 -*-
import csv, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import validate_v36 as v
from v34_reference import SIGNATURE, VIEW_NAMES

class ValidationV36Tests(unittest.TestCase):
    def test_crop_matches_v34(self):
        im=Image.new('RGB',(420,225),color='gray')
        a=v.crop(im,'center72');b=v.crop(im,'upper_right');c=v.crop(im,'center90')
        self.assertEqual(a.size,b.size)
        self.assertEqual(c.size,(378,202))
        self.assertEqual(v.crop(im,'global').size,(420,225))

    def test_wilson_bounds(self):
        a,b=v.wilson(539,558)
        self.assertLess(a,539/558)
        self.assertGreater(b,539/558)
        self.assertLess(a,b)

    def test_score(self):
        y=np.array([0,1,2,3,0,1,2,3]);p=np.eye(4)[y].astype(float)
        out=v.score(y,p)
        self.assertEqual(out['correct'],8)
        self.assertEqual(out['confusion_true_rows'],(np.eye(4)*2).astype(int).tolist())

    def test_exact_dev_test_collision_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'data/test/normal').mkdir(parents=True)
            (d/'data/train/normal').mkdir(parents=True)
            im=Image.new('RGB',(64,64),color='black')
            im.save(d/'data/train/normal/one.png')
            im.save(d/'data/test/normal/other.png')
            a=SimpleNamespace(data_root=d/'data',out=d/'out')
            with self.assertRaisesRegex(ValueError,'EXACT DEV-TEST duplicate'):
                v.build_test_inventory(a,['train/normal/one.png'],expected=1)
            report=json.loads((d/'out/holdout_overlap_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(report['exact_duplicate_count'],1)

    def test_external_manifest_rejects_other_classes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            Image.new('RGB',(30,30),color='black').save(d/'slice.png')
            with (d/'manifest.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['image_path','patient_id','class_name','cohort'])
                w.writerow(['slice.png','patient001','NOR','StudyX'])
            with self.assertRaisesRegex(ValueError,'ADC/SCC ONLY'):
                v.parse_external(SimpleNamespace(manifest=d/'manifest.csv'))

    def test_plan_and_2class_rule(self):
        self.assertEqual(SIGNATURE,'522710c76a60df73d6776ae8a05e75d46c943dbc006215bf81d045a49ed3dae2')
        self.assertEqual(VIEW_NAMES[0],'global')
        self.assertEqual(v.normalized_label('large.cell.carcinoma_left.hilum'),1)

if __name__=='__main__':unittest.main()
