"""Verify the asset-discovery objective independently of GPU generation."""
import unittest
import numpy as np
from sample_assets import exhaustive_bank,build_item,summarize,geometry


class AssetsTest(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='x',suite='gsm8k',task='gsm8k',question='2+2?',split='test',target='4')

    def records(self,values):
        return {('x',name):dict(id='x',action=name,reward=value,text='proof') for name,value in values.items()}

    def test_keeps_every_success_and_failure(self):
        item=build_item(self.row,self.records({'none':0,'a':1,'b':1,'c':0}),['none','a','b','c'])
        self.assertEqual(set(item['successful_directions']),{'a','b'})
        self.assertEqual(set(item['failed_directions']),{'none','c'})
        self.assertTrue(item['rescued']);self.assertEqual(item['status'],'complete')
        self.assertEqual(item['evidence']['a']['text'],'proof')

    def test_worse_steering_cannot_erase_baseline(self):
        item=build_item(self.row,self.records({'none':1,'bad':0}),['none','bad'])
        self.assertEqual(item['best_reward'],1);self.assertEqual(item['best_action'],'none')
        self.assertTrue(item['solved']);self.assertFalse(item['rescued'])

    def test_unsolved_and_missing_are_distinct(self):
        item=build_item(self.row,self.records({'none':0,'bad':0}),['none','bad','pending'])
        self.assertFalse(item['solved']);self.assertEqual(item['status'],'partial')
        self.assertEqual(item['missing_actions'],['pending'])
        self.assertFalse(summarize([item])['all_complete'])

    def test_truthful_probability_and_correctness_separate(self):
        row=dict(self.row,suite='truthfulqa',targets=dict(labels=[0,1],choices=['wrong','right']))
        records=self.records({'none':.1,'better':.4,'solves':.7})
        for r in records.values():r['choice_probabilities']=[1-r['reward'],r['reward']]
        item=build_item(row,records,['none','better','solves'])
        self.assertEqual(item['successful_directions'],['solves'])
        self.assertEqual(set(item['improving_directions']),{'better','solves'})

    def test_writing_continuous_improvements(self):
        row=dict(self.row,suite='writingbench')
        item=build_item(row,self.records({'none':.5,'equal':.5,'better':.6,'worse':.4}),['none','equal','better','worse'])
        self.assertEqual(item['successful_directions'],['better']);self.assertIsNone(item['solved'])

    def test_full_declared_bank_signed_doses_norm_bounds(self):
        geo=geometry();bank=exhaustive_bank(geo)
        self.assertEqual(len(bank),37);self.assertEqual(len({a['id'] for a in bank}),37)
        self.assertEqual(bank[0]['id'],'none')
        for i in range(9):
            settings=[a for a in bank if a.get('concept')==bank[1+2*i]['concept']]
            self.assertEqual({a['requested_strength'] for a in settings},{-1,-.5,.5,1})
        for a in bank:
            np.testing.assert_allclose(np.asarray(a['coefficients'])@geo['directions'],a['shift'])
            self.assertLessEqual(a['norm'],geo['scale']+1e-9)


if __name__=='__main__':unittest.main()
