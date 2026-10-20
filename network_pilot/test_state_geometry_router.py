import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

import state_geometry_models as models
import state_geometry_router as experiment


class StateGeometryTests(unittest.TestCase):
    def test_oof_scores_match_fold_local_refits(self):
        rng=np.random.default_rng(25)
        x=rng.normal(size=(15,4));y=(rng.uniform(size=(15,3))>.5).astype(float)
        geometry=models.action_kernel(np.array([[0.,0.],[1.,0.],[-1.,0.]]))
        folds=models.three_folds(len(x))
        for kind in ['multi','multi_rbf']:
            best,_,oof=models.select({kind:x},y,folds,geometry,True)
            for fold in range(3):
                tr=folds!=fold;va=~tr
                fitted=models.fit_reward(x[tr],y[tr],best['alpha'],geometry,best['strength'],
                          kernel='rbf' if kind.endswith('_rbf') else 'linear')
                np.testing.assert_allclose(oof[va],models.predict(fitted,x[va]),rtol=1e-9,atol=1e-9)

    def test_direction_score_generalizes_and_gate_can_preserve_noop(self):
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([[0,0,1],[0,0,1],[0,1,0],[0,1,0]],dtype=float)
        geometry=models.action_kernel(np.array([[0.,0.],[1.,0.],[-1.,0.]]))
        model=models.fit_reward(x,y,.01,geometry,.03)
        scores=models.predict(model,np.array([[-1.5],[1.5]]))
        np.testing.assert_array_equal(models.choose(scores),[2,1])
        np.testing.assert_array_equal(models.choose(scores,float('inf')),[0,0])

    def test_capture_excludes_padding_and_does_not_read_answers(self):
        import torch
        class Batch(dict):
            def __getattr__(self,key):return self[key]
            def to(self,device):return self
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                self.texts=texts
                return Batch(input_ids=torch.tensor([[100,1,3],[2,4,6]]),
                             attention_mask=torch.tensor([[0,1,1],[1,1,1]]))
        class Block(torch.nn.Module):
            def __init__(self,add):super().__init__();self.add=add
            def forward(self,x):return x+1 if self.add else x*2
        class Model:
            def __init__(self):
                self.model=SimpleNamespace(language_model=SimpleNamespace(layers=[Block(True),Block(False)]))
            def __call__(self,input_ids,attention_mask,**kwargs):
                h=input_ids.float()[:,:,None].repeat(1,1,2)
                for block in self.model.language_model.layers:h=block(h)
                return SimpleNamespace()
        e=SimpleNamespace(torch=torch,device='cpu',layer=0,model=Model(),tokenizer=Tokenizer(),prefix=lambda q:q)
        rows=[dict(id=f'q{i}',suite='ifeval',question=q,target='DO_NOT_SEND') for i,q in enumerate(['A','B'])]
        captured,meta=experiment.capture_batch(e,rows)
        self.assertEqual(e.tokenizer.texts,['A','B'])
        np.testing.assert_array_equal(captured['steer_last'],[[4,4],[7,7]])
        np.testing.assert_array_equal(captured['final_last'],[[8,8],[14,14]])
        np.testing.assert_array_equal(captured['final_mean'],[[6,6],[10,10]])
        self.assertEqual(meta['generated_tokens'],0)

    def test_changing_test_answers_cannot_change_fit_or_frozen_choices(self):
        from fit_asset_router import vectorizer
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);d=out/'gsm8k';d.mkdir()
            rows=[dict(id=f'q{i}',suite='gsm8k',question=f'item {i}',target='1',
                       split='train' if i<12 else 'test') for i in range(18)]
            p=dict(records=rows,actions_sha256='fixture')
            experiment.core.save(d/'protocol.json',p)
            actions=np.array(['none','positive','negative'])
            np.savez(d/'actions.npz',ids=actions,shifts=np.array([[0.,0.],[1.,0.],[-1.,0.]]))
            sign=np.array([-1.,1.]*9)
            y=np.zeros((12,3));y[np.arange(12),np.where(sign[:12]>0,1,2)]=1
            np.savez(d/'train_matrix.npz',reward=y)
            transform=vectorizer();transform.fit(['fixture text'])
            joblib.dump(dict(transform=transform,w=np.zeros((len(transform.vocabulary_),3)),
                             bias=np.zeros(3),actions=actions),d/'router-set.joblib')
            feature=np.stack([sign,np.ones(18)],axis=1).astype(np.float32)
            features={k:feature.copy() for k in ['steer_last','final_last','final_mean']}
            with patch.object(experiment,'load_features',return_value=(features,[])):
                experiment.fit_suite(out,'gsm8k')
                decisions=(d/'frozen_test_actions.json').read_bytes()
                first=joblib.load(d/'router-geometry.joblib')
                for r in p['records']:
                    if r['split']=='test':r['target']='ALTERED ANSWER'
                experiment.core.save(d/'protocol.json',p)
                experiment.fit_suite(out,'gsm8k')
            second=joblib.load(d/'router-geometry.joblib')
            self.assertEqual((d/'frozen_test_actions.json').read_bytes(),decisions)
            np.testing.assert_array_equal(first['model']['coef'],second['model']['coef'])


if __name__=='__main__':unittest.main()
