"""Check prompt isolation and test-label isolation for the replicated design."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import gsm_router_large as core
from fit_asset_router import fit


class ReplicationTests(unittest.TestCase):
    def test_official_instruction_scoring_after_gpu_import_path_changes(self):
        import sys
        with patch.object(sys,'path',[str(core.LAB)]+sys.path):
            row=dict(id='ifeval:fixture',suite='ifeval',question='Respond in uppercase.',source_index=0,
                     instruction_id_list=['change_case:english_capital'],kwargs=[{}])
            good=core.grade('THIS IS A CORRECT UPPERCASE ENGLISH SENTENCE.',row)
            bad=core.grade('This is a lowercase English sentence.',row)
        self.assertTrue(good['correct'])
        self.assertFalse(bad['correct'])

    def test_conflicting_canonical_labels_are_not_reported_as_full_fit(self):
        x=csr_matrix([[1.,0.],[1.,0.]])
        valid=np.array([[True,False],[False,True]])
        trace=[]
        w,bias=fit(x,valid,trace,require_full_fit=False)
        selected=np.asarray(x@w+bias).argmax(1)
        self.assertEqual(int(valid[np.arange(2),selected].sum()),1)
        with self.assertRaises(AssertionError):fit(x,valid,[])

    def test_instruction_prompt_preserves_exact_requested_format(self):
        question='Respond with exactly two words and no punctuation.'
        self.assertEqual(core.question(dict(suite='ifeval',question=question)),question)
        self.assertIn('FINAL:',core.question(dict(suite='bbh',question='True or False?')))

    def test_test_answers_cannot_change_fitted_router_or_fixed_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            rows=[dict(id='t0',suite='bbh',question='red apples',split='train',target='True'),
                  dict(id='t1',suite='bbh',question='blue boats',split='train',target='False'),
                  dict(id='t2',suite='bbh',question='uncovered item',split='train',target='True'),
                  dict(id='q0',suite='bbh',question='red apples in a box',split='test',target='True'),
                  dict(id='q1',suite='bbh',question='blue boats at sea',split='test',target='False')]
            protocol=dict(n_train=3,shards=1,records=rows)
            core.save(d/'protocol.json',protocol)
            np.savez(d/'actions.npz',ids=np.array(['none','steer']),shifts=np.zeros((2,3)))
            train=d/'train/shard-0';train.mkdir(parents=True)
            core.save(train/'complete.json',dict(records=6))
            for i,rewards in enumerate([[1,0],[0,1],[0,0]]):
                for action,reward in zip(['none','steer'],rewards):
                    core.append(train/'records.jsonl',dict(id=f't{i}',action=action,reward=reward))
            core.fit_routers(d)
            decisions=(d/'frozen_test_actions.json').read_bytes()
            summary=json.loads((d/'training_summary.json').read_text())
            self.assertEqual(summary['fixed_action'],'none')
            self.assertEqual(summary['training']['set']['fitted'],2)
            self.assertEqual(summary['training']['set']['train_total'],3)
            for r in protocol['records']:
                if r['split']=='test':r['target']='ALTERED TEST ANSWER'
            core.save(d/'protocol.json',protocol)
            core.fit_routers(d)
            self.assertEqual((d/'frozen_test_actions.json').read_bytes(),decisions)


if __name__=='__main__':unittest.main()
