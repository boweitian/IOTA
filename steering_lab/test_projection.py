"""Projection constraints preserve orthogonal edits and bound correlated edits."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from block_presets import FAITH_AXIS
from studio import AXES, PRESETS, mixed_shift, profile_for, strength_regularization, validate_profile
import test_agent

# Also consumed by the browser parity check. Each vector represents its own 1x.
CASES = [
    dict(name='orthogonal', vectors=[[1.,0.],[0.,1.]], coefficients=[1.,1.], factor=1.),
    dict(name='aligned', vectors=[[1.,0.],[1.,0.]], coefficients=[2.,1.], factor=2/3),
    dict(name='negative', vectors=[[1.,0.],[1.,0.]], coefficients=[-2.,-1.], factor=2/3),
    dict(name='opposite poles', vectors=[[1.,0.],[-1.,0.]], coefficients=[2.,-1.], factor=2/3),
    dict(name='subunit maximum', vectors=[[1.,0.],[1.,0.]], coefficients=[.4,.4], factor=.5),
    dict(name='unequal unit norms', vectors=[[2.,0.],[1.,0.]], coefficients=[1.,1.], factor=1/3),
    dict(name='partial cancellation', vectors=[[1.,0.],[1.,0.]], coefficients=[1.,-.5], factor=1.),
    dict(name='exact cancellation', vectors=[[1.,0.],[1.,0.]], coefficients=[1.,-1.], factor=1.),
    dict(name='inactive direction', vectors=[[1.,0.],[0.,1.],[math.sqrt(.5),math.sqrt(.5)]], coefficients=[1.,1.,0.], factor=1.),
    dict(name='no intervention', vectors=[[1.,0.],[0.,1.]], coefficients=[0.,0.], factor=1.),
]

def case_profile(case):
    profile=profile_for(PRESETS[1])
    profile['coefficients']={key:0. for key in AXES}
    directions=np.zeros((len(AXES),len(case['vectors'][0])))
    for i,(vector,value) in enumerate(zip(case['vectors'],case['coefficients'])):
        directions[i]=vector;profile['coefficients'][AXES[i]]=value
    return profile,directions

class ProjectionTests(unittest.TestCase):
    def test_known_geometries_are_bounded_with_minimal_common_scaling(self):
        for case in CASES:
            with self.subTest(case=case['name']):
                profile,directions=case_profile(case)
                shift,info=mixed_shift(profile,directions,1.,regularization='direction_projection')
                raw=sum((a*np.array(d) for a,d in zip(case['coefficients'],case['vectors'])))
                self.assertAlmostEqual(info['cap_factor'],case['factor'])
                np.testing.assert_allclose(shift,raw*case['factor'],atol=1e-12)
                limit=max(map(abs,case['coefficients']))
                self.assertEqual(info['projection_limit'],limit)
                for a,d in zip(case['coefficients'],case['vectors']):
                    if a:
                        projection=float(np.dot(shift,d)/np.dot(d,d))
                        self.assertLessEqual(abs(projection),limit+1e-12)
                if case['factor']<1:
                    self.assertAlmostEqual(info['max_projection'],limit)
                self.assertIsNone(info['norm_cap'])
        profile,directions=case_profile(CASES[0])
        shift,_=mixed_shift(profile,directions,1.,regularization='direction_projection')
        self.assertGreater(np.linalg.norm(shift),1.)

    def test_rotating_geometry_does_not_change_the_scale(self):
        rng=np.random.default_rng(1234)
        for _ in range(40):
            directions=rng.normal(size=(len(AXES),5))
            profile=profile_for(PRESETS[1])
            profile['coefficients']=dict(zip(AXES,rng.uniform(-2,2,len(AXES))))
            shift,info=mixed_shift(profile,directions,1.,regularization='direction_projection')
            rotation,_=np.linalg.qr(rng.normal(size=(5,5)))
            rotated,other=mixed_shift(profile,directions@rotation,1.,regularization='direction_projection')
            np.testing.assert_allclose(rotated,shift@rotation,atol=1e-12)
            self.assertAlmostEqual(info['cap_factor'],other['cap_factor'])
            independent=directions@shift/np.einsum('ij,ij->i',directions,directions)
            self.assertLessEqual(np.max(np.abs(independent)),max(map(abs,profile['coefficients'].values()))+1e-12)

    def test_custom_and_full_position_shifts_share_the_same_scale(self):
        fixture=test_agent.AgentTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        studio=fixture.studio;profile=fixture.profile
        profile['coefficients']={key:0. for key in AXES}
        profile['coefficients'][AXES[0]]=2.
        profile['custom_axes']=[dict(FAITH_AXIS)]
        profile['coefficients'][FAITH_AXIS['id']]=1.
        studio.directions=np.eye(len(AXES),4)*studio.cap
        studio.build_custom=Mock(return_value={'unit':np.array([1.,0.,0.,0.])})
        with patch('studio.strength_regularization', return_value='direction_projection'):
            result=studio.chat(validate_profile(profile),fixture.messages,fixture.binding)
        args,kw=studio.engine.generate.call_args
        np.testing.assert_allclose(args[1][0],[.4,0.,0.,0.])
        np.testing.assert_allclose(kw['prefill_shifts'][0],[.2*2/3,0.,0.,0.])
        self.assertAlmostEqual(result['intervention']['cap_factor'],2/3)
        self.assertAlmostEqual(result['intervention']['projections'][FAITH_AXIS['id']],2.)
        self.assertEqual(result['intervention']['effective_coefficients'][AXES[0]],4/3)

    def test_policy_can_be_changed_without_reloading_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json'
            with patch('studio.STRENGTH_POLICY',path):
                for mode in ['direction_projection','none','norm']:
                    path.write_text(json.dumps(dict(regularization=mode)))
                    self.assertEqual(strength_regularization(),mode)

    def test_active_zero_vector_is_rejected_but_disabled_zero_is_ignored(self):
        profile,directions=case_profile(CASES[0])
        directions[0]=0
        with self.assertRaises(ValueError):mixed_shift(profile,directions,1.,regularization='direction_projection')
        profile['coefficients'][AXES[0]]=0.
        shift,_=mixed_shift(profile,directions,1.,regularization='direction_projection')
        np.testing.assert_allclose(shift,[0.,1.])

if __name__=='__main__':unittest.main()
