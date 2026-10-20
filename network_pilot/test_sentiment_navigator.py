import unittest
import numpy as np
from sentiment_prepare import read_trees,balanced
from sentiment_models import folds,fit_team,fit_policy,predict,train
from sentiment_navigator import bank

class SentimentTests(unittest.TestCase):
    def test_whole_sentence_roots_not_subphrases(self):
        text='(3 (2 A) (4 (3 lovely) (2 film)))\n(1 (2 A) (1 mess))\n'
        rows=read_trees(text,'train')
        self.assertEqual([(r['text'],r['label']) for r in rows],[('A lovely film',3),('A mess',1)])

    def test_stratified_folds_cover_only_training_rows(self):
        y=np.repeat(np.arange(5),14);split=folds(y)
        self.assertEqual(set(split),set(range(5)))
        for fold in range(5):
            counts=np.bincount(y[split==fold],minlength=5)
            self.assertTrue(np.all((counts==2)|(counts==3)))
            self.assertFalse(set(np.flatnonzero(split==fold))&set(np.flatnonzero(split!=fold)))

    def test_team_and_router_produce_valid_probabilities(self):
        rng=np.random.default_rng(1);p=rng.dirichlet(np.ones(5),size=(70,9));x=rng.normal(size=(70,9));y=np.repeat(np.arange(5),14)
        for name in ['baseline','best_single','team_3','team_5','router_3_10','uniform_3']:
            policy=fit_policy(name,p,x,y,[0,1,2]);result=predict(policy,p,x)
            np.testing.assert_allclose(result.sum(1),1.)
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(len(set(policy['members'])),len(policy['members']))
            self.assertAlmostEqual(sum(policy['weights']),1.)

    def test_identical_agents_do_not_claim_ensemble_gain(self):
        p=np.full((70,9,5),.2);x=np.zeros((70,9));y=np.repeat(np.arange(5),14)
        result=train(p,x,y,[0,1,2])
        self.assertEqual(result['selected'],'baseline')
        self.assertTrue(all(row['correct']==14 for row in result['cv']))

    def test_candidate_mixtures_follow_projection_constraints(self):
        directions=np.eye(9)
        actions=bank(directions,1.)
        self.assertEqual(len(actions),49)
        for action in actions:
            info=action['intervention']
            self.assertLessEqual(info['max_projection'],info['projection_limit']+1e-12)
        self.assertEqual(actions[0]['id'],'baseline')

if __name__=='__main__':unittest.main()
