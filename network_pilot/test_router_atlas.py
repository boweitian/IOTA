import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import build_router_atlas as atlas
import router_direction_scan as scan


class FakeEngine:
    meta={'device':'test'}
    def __init__(self):self.calls=[]
    def generate(self,questions,shifts,max_new_tokens):
        self.calls.append((questions,np.array(shifts)))
        return [dict(text='FINAL: 1',token_ids=[1],new_tokens=1,truncated=False,
                     batch_seconds=.01,batch_size=len(questions)) for _ in questions]


class AtlasTests(unittest.TestCase):
    def test_alignment_is_exact_and_intervention_stays_on_x(self):
        rng=np.random.default_rng(56);training=rng.normal(size=(60,12))
        direction=rng.normal(size=12);direction*=.5/np.linalg.norm(direction)
        p=atlas.aligned_pca(training,direction)
        np.testing.assert_allclose(p['axes'][0],direction/np.linalg.norm(direction),atol=1e-12)
        np.testing.assert_allclose(p['axes']@p['axes'].T,np.eye(3),atol=1e-12)
        point=rng.normal(size=12)
        before=(point-p['mean'])@p['axes'].T;after=(point+direction-p['mean'])@p['axes'].T
        np.testing.assert_allclose(after-before,[.5,0,0],atol=1e-12)
        heldout=rng.normal(size=(7,12))*1000
        projected=(heldout-p['mean'])@p['axes'].T
        self.assertEqual(projected.shape,(7,3))
        repeated=atlas.aligned_pca(training,direction)
        np.testing.assert_allclose(repeated['axes'],p['axes'],atol=1e-12)

    def test_best_representative_keeps_ties_and_no_success_is_not_invented(self):
        chosen,valid=atlas.best_representative([1,0,1,1],[3,100,8,5])
        self.assertEqual(chosen,2);self.assertEqual(valid,[0,2,3])
        self.assertEqual(atlas.best_representative([0,0,0],[3,5,9]),(None,[]))
        self.assertEqual(atlas.best_representative([-1,0,-1],[3,5,9]),(None,[]))

    def test_per_question_reference_axes_keep_sign_strength_and_train_only_fit(self):
        rng=np.random.default_rng(5);states=rng.normal(size=(35,12));train=np.arange(35)<25
        vector=rng.normal(size=12);shifts=np.stack([vector*0,vector,-vector,vector*2])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'toy').mkdir()
            bank=atlas.projection_bank(root,'toy',states,train,shifts)
            self.assertEqual(len(bank['entries']),1)
            self.assertEqual(bank['mapping'],[-1,0,0,0])
            entry=bank['entries'][0]
            for j in [1,2,3]:
                sign=bank['signs'][j];coordinates=np.array(entry['coordinates']);coordinates[:,0]*=sign
                expected=(states-states[train].mean(0))@(shifts[j]/np.linalg.norm(shifts[j]))
                np.testing.assert_allclose(coordinates[:,0],expected,atol=6e-8)
                projected=np.array(entry['vectors'])[j];projected[0]*=sign
                np.testing.assert_allclose(projected,[np.linalg.norm(shifts[j]),0,0],atol=1e-8)
            cached=atlas.projection_bank(root,'toy',states,train,shifts)
            self.assertEqual(cached['entries'],bank['entries'])

    def test_partial_scan_resume_preserves_entire_original_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);d=out/'gsm8k';d.mkdir()
            rows=[dict(id=f'q{i}',suite='gsm8k',question=f'one? {i}',target='1',task='gsm8k',
                       split='test',eval_group='historical') for i in range(3)]
            scan.core.save(d/'protocol.json',dict(records=rows,shards=1,batch_size=2,max_new_tokens=16,model={}))
            np.savez(d/'actions.npz',ids=np.array(['none','shift']),shifts=np.array([[0.,0.],[1.,0.]]))
            folder=d/'scan/shard-0';folder.mkdir(parents=True)
            scan.core.append(folder/'records.jsonl',dict(id='q0',action='none'))
            e=FakeEngine()
            with patch.object(scan.core,'engine',return_value=e):scan.sweep(out,'gsm8k',0)
            self.assertEqual(len(e.calls[0][0]),2)
            records=scan.core.read(folder/'records.jsonl')
            self.assertEqual(len(records),6)
            self.assertEqual(len({(r['id'],r['action']) for r in records}),6)


if __name__=='__main__':unittest.main()
