import unittest
from unittest.mock import patch

import numpy as np

import fixed_advantage_models as m


class AdvantageTests(unittest.TestCase):
    def test_repair_damage_targets_and_fixed_column(self):
        y=np.array([[0.,1.,1.],[1.,0.,1.]])
        target=m.targets(y,1)
        np.testing.assert_equal(target[:,:3],[[0,0,0],[1,0,1]])
        np.testing.assert_equal(target[:,3:],[[1,0,0],[0,0,0]])

    def test_fixed_fallback_damage_penalty_and_train_action_mask(self):
        heads=np.array([[.9,0,.6,.5,0,.05],[0,0,0,0,0,0.]])
        chosen,_=m.choose(heads,1,[10,20,1],penalty=1.,top=3)
        np.testing.assert_equal(chosen,[2,1])
        conservative,_=m.choose(heads,1,[10,20,1],penalty=2.,top=3,threshold=.55)
        np.testing.assert_equal(conservative,[1,1])
        masked,_=m.choose(heads,1,[10,20,1],penalty=2.,top=1)
        np.testing.assert_equal(masked,[1,1])
        always_fixed,_=m.choose(heads,1,[10,20,1],threshold=float('inf'))
        np.testing.assert_equal(always_fixed,[1,1])

    def test_model_uses_public_question_not_answers_and_round_trip(self):
        import pickle
        rows=[dict(suite='test',task='arithmetic',question=f'Question number {i}: add apples and pears.') for i in range(18)]
        rng=np.random.default_rng(1);h={k:rng.normal(size=(18,8)) for k in ['steer_last','final_last','final_mean']}
        y=rng.integers(0,2,size=(18,3)).astype(float)
        config=dict(kind='hybrid_rbf',alpha=1.,penalty=2.,threshold=.03,top=3)
        fitted=m.fit(rows,h,y,config);chosen,score=m.predict(fitted,rows,h)
        changed=[dict(r,target='WRONG NEW ANSWER',id='fake',response='irrelevant') for r in rows]
        again,other=m.predict(pickle.loads(pickle.dumps(fitted)),changed,h)
        np.testing.assert_equal(chosen,again);np.testing.assert_allclose(score,other)
        self.assertEqual(fitted['fixed'],int(y.sum(0).argmax()))

    def test_fold_vocabularies_exclude_validation_only_tokens(self):
        rows=[dict(suite='s',task='a' if i<9 else 'b',question=f'Common sample UNIQUE{i}') for i in range(18)]
        rng=np.random.default_rng(2);h={k:rng.normal(size=(18,5)) for k in ['steer_last','final_last','final_mean']}
        y=rng.integers(0,2,size=(18,3)).astype(float);folds=m.folds_for(rows)
        fit_original=m.fit_features;seen=[]
        def observe(training,hidden):
            fitted,features=fit_original(training,hidden)
            vocab=fitted['word'].vocabulary_;ids={r['question'].split()[-1] for r in training}
            for i in range(18):self.assertEqual(f'UNIQUE{i}' in vocab,f'UNIQUE{i}' in ids)
            seen.append(len(training));return fitted,features
        with patch.object(m,'fit_features',side_effect=observe),patch.object(m,'ALPHAS',(1.,)),patch.object(m,'KINDS',('text',)),patch.object(m,'PENALTIES',(1.,)),patch.object(m,'TOP_COUNTS',(3,)):
            selected,trials,_=m.select(rows,h,y)
        self.assertEqual(seen,[12,12,12]);self.assertEqual(len(folds),18)
        self.assertGreaterEqual(selected['correct'],trials[0]['fixed_correct'])


if __name__=='__main__':unittest.main()
