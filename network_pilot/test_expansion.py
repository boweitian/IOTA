"""Geometry reconstruction and exact preservation tests using synthetic activations."""
import unittest
import numpy as np
from expanded_geometry import fit
from expanded_concepts import ADDITIONS,CONCEPTS,probes
from expanded_assets import bank_for
from sample_assets import geometry,exhaustive_bank

class ExpansionTests(unittest.TestCase):
    def test_geometry_preserves_original_vectors_and_excludes_diagnostics(self):
        rng=np.random.default_rng(41);old={k:rng.normal(size=(9,32)).astype(np.float32) for k in ['raw','unit','directions']}
        old['unit']/=np.linalg.norm(old['unit'],axis=1,keepdims=True)
        examples=probes();pooled=rng.normal(size=(360,32)).astype(np.float32)
        fitted,diagnostics=fit(pooled,examples,old,.507)
        for k in old:np.testing.assert_array_equal(fitted[k][:9],old[k])
        changed=pooled.copy()
        for i,p in enumerate(examples):
            if p['context']>=8:changed[i]*=8
        refitted,_=fit(changed,examples,old,.507)
        np.testing.assert_array_equal(refitted['directions'],fitted['directions'])
        np.testing.assert_allclose(np.linalg.norm(fitted['directions'][9:],axis=1),.507,rtol=1e-6)
        self.assertEqual(fitted['cosine'].shape,(27,27));self.assertEqual(fitted['coordinates'].shape,(27,3))
        self.assertEqual(len(diagnostics),18)

    def test_bank_keeps_all_old_physical_shifts_and_adds_every_signed_dose(self):
        old=geometry();geo=dict(old,directions=np.concatenate([old['directions'],np.tile(old['directions'][:1],(18,1))]))
        expanded=bank_for(geo);prior=exhaustive_bank(old)
        self.assertEqual(len(expanded),109)
        for before,after in zip(prior,expanded):
            self.assertEqual(before['id'],after['id']);np.testing.assert_array_equal(before['shift'],after['shift'])
            self.assertEqual(after['coefficients'][:9],before['coefficients']);self.assertEqual(after['coefficients'][9:],[0.]*18)
        for i in range(9,27):
            found=[a for a in expanded if a.get('concept')==CONCEPTS[i]['id']]
            self.assertEqual({a['requested_strength'] for a in found},{-1.,-.5,.5,1.})
            self.assertTrue(all(a['norm']<=old['scale']+1e-12 for a in found))

    def test_contrasts_are_disjoint_unique_and_complete(self):
        self.assertEqual(len({c['id'] for c in CONCEPTS}),27)
        self.assertEqual(len(probes()),360)
        self.assertTrue(all(c['positive']!=c['negative'] for c in ADDITIONS))
        self.assertEqual({p['split'] for p in probes()},{'fit','diagnostic'})

if __name__=='__main__':unittest.main()
