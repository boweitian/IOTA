import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import gsm_router_large as experiment


class FakeEngine:
    meta={'device':'test'}

    def __init__(self):
        self.calls=[]

    def generate(self,questions,shifts,max_new_tokens):
        self.calls.append((questions,np.stack(shifts),max_new_tokens))
        return [dict(text='FINAL: 1',token_ids=[1],new_tokens=1,truncated=False,
                     batch_seconds=.01,batch_size=len(questions)) for _ in questions]


class BatchTests(unittest.TestCase):
    def fixture(self,out):
        rows=[dict(id=f'q{i}',split='train' if i<4 else 'test',question=f'What is one? item {i}',
                   task='gsm8k',target='1') for i in range(6)]
        np.savez(out/'actions.npz',ids=np.array(['none','a']),shifts=np.array([[0.,0.],[1.,0.]]))
        experiment.save(out/'protocol.json',dict(shards=1,batch_size=2,max_new_tokens=64,
             records=rows,model={},actions_sha256=experiment.sha(out/'actions.npz')))

    def test_resuming_partial_batch_preserves_original_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.fixture(out)
            shard=out/'train/shard-0';shard.mkdir(parents=True)
            experiment.append(shard/'records.jsonl',dict(id='q0',action='none'))
            engine=FakeEngine()
            with patch.object(experiment,'engine',return_value=engine):experiment.train_measure(out,0)
            self.assertEqual(len(engine.calls),4)
            self.assertEqual(len(engine.calls[0][0]),2)
            records=experiment.read(shard/'records.jsonl')
            self.assertEqual(len(records),8)
            self.assertEqual(len({(r['id'],r['action']) for r in records}),8)

    def test_test_methods_share_cohorts_and_use_frozen_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);self.fixture(out)
            decisions=[dict(id=f'q{i}',baseline='none',fixed='a',set='a',majority='none' if i==4 else 'a')
                       for i in [4,5]]
            experiment.save(out/'frozen_test_actions.json',decisions)
            experiment.save(out/'training_summary.json',dict(frozen_test_actions_sha256=experiment.sha(out/'frozen_test_actions.json')))
            engine=FakeEngine()
            with patch.object(experiment,'engine',return_value=engine):experiment.test_measure(out,0)
            self.assertEqual(len(engine.calls),4)
            self.assertTrue(all(c[0]==engine.calls[0][0] for c in engine.calls))
            np.testing.assert_equal(engine.calls[0][1],np.zeros((2,2)))
            np.testing.assert_equal(engine.calls[-1][1],np.array([[0,0],[1,0]]))


if __name__=='__main__':unittest.main()
