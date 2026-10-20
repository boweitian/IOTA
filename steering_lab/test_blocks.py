"""Actual strength budgets and mixed last/all-position execution, not UI labels."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
from block_presets import FAITH_AXIS, faith_payload
from studio import AXES, PRESETS, CharacterStudio, profile_for, validate_profile, validate_custom_axes
import test_agent


class BlockTests(unittest.TestCase):
    def setUp(self):
        test_agent.AgentTests.setUp(self)

    def test_paused_cap_preserves_coefficients_and_full_position_component(self):
        self.profile['coefficients']={k:0. for k in AXES}
        self.profile['coefficients'][AXES[0]]=-2.
        self.profile['custom_axes']=[dict(FAITH_AXIS)]
        self.profile['coefficients'][FAITH_AXIS['id']]=2.
        self.studio.directions=np.eye(len(AXES),4)*self.studio.cap
        self.studio.build_custom=Mock(return_value={'unit':np.array([0.,1.,0.,0.])})
        clean=validate_profile(self.profile)
        with patch('studio.strength_regularization', return_value='none'):
            result=self.studio.chat(clean,self.messages,self.binding)
        args,kw=self.studio.engine.generate.call_args
        np.testing.assert_allclose(args[1][0],[-.4,.4,0.,0.])
        np.testing.assert_allclose(kw['prefill_shifts'][0],[0.,.4,0.,0.])
        info=result['intervention']
        self.assertEqual(info['cap_factor'],1.)
        self.assertEqual(info['norm'],info['raw_norm'])
        self.assertGreater(info['norm'],info['cap'])
        self.assertFalse(info['total_strength_cap_enabled'])
        # Re-enabling the policy restores proportional scaling for the same profile.
        shift,info,full=self.studio.intervention(clean)
        self.assertLess(info['cap_factor'],1.)
        self.assertAlmostEqual(np.linalg.norm(shift),info['cap'])

    def test_two_times_strength_is_not_clipped_to_the_old_cap(self):
        self.profile['coefficients']={k:0. for k in AXES}
        self.profile['coefficients'][AXES[0]]=2.
        self.studio.directions=np.eye(len(AXES),4)*self.studio.cap
        clean=validate_profile(self.profile)
        shift,info,full=self.studio.intervention(clean)
        self.assertAlmostEqual(np.linalg.norm(shift),2*self.studio.cap)
        self.assertEqual(info['strength_multiplier'],2)
        self.assertIsNone(full)
        self.profile['coefficients'][AXES[0]]=2.01
        with self.assertRaises(ValueError):validate_profile(self.profile)

    def test_faith_block_and_regular_block_use_distinct_prefill_shifts(self):
        self.profile['coefficients']={k:0. for k in AXES}
        self.profile['coefficients'][AXES[0]]=.5
        self.profile['custom_axes']=[dict(FAITH_AXIS)]
        self.profile['coefficients'][FAITH_AXIS['id']]=2.
        self.studio.directions=np.eye(len(AXES),4)*self.studio.cap
        self.studio.build_custom=Mock(return_value={'unit':np.array([0.,1.,0.,0.])})
        clean=validate_profile(self.profile);shift,info,full=self.studio.intervention(clean)
        self.assertLessEqual(np.linalg.norm(shift),2*self.studio.cap+1e-12)
        np.testing.assert_allclose(full,[0.,.4*info['cap_factor'],0.,0.])
        self.assertGreater(shift[0],0)
        self.assertEqual(info['full_position_axes'],[FAITH_AXIS['id']])
        self.studio.chat(clean,self.messages,self.binding)
        args,kw=self.studio.engine.generate.call_args
        np.testing.assert_allclose(kw['prefill_shifts'][0],full)
        np.testing.assert_allclose(args[1][0],shift)

    def test_experimental_source_is_not_spoofable(self):
        self.assertEqual(validate_custom_axes([dict(FAITH_AXIS)])[0],FAITH_AXIS)
        for field,value in [('source','unknown'),('id','custom-fake'),('positive','a different recipe')]:
            with self.assertRaises(ValueError):validate_custom_axes([dict(FAITH_AXIS,**{field:value})])

    def test_packaged_faith_direction_is_one_times_at_unit_coefficient(self):
        import json
        from pathlib import Path
        atlas=json.loads((Path(__file__).resolve().parent/'runs/qwen35-9b-studio-v2/atlas.json').read_text())
        result=faith_payload(atlas,'test')
        self.assertAlmostEqual(np.linalg.norm(result['native_direction']),atlas['scale'],places=5)
        self.assertEqual(result['positions'],'all')
        self.assertEqual(result['recommended_strength'],2.)
        with self.assertRaises(ValueError):faith_payload(dict(atlas,model=dict(atlas['model'],revision='wrong')),'test')


class EngineHookTests(unittest.TestCase):
    def test_full_position_component_is_not_double_counted_on_last_token(self):
        import torch
        from engine import Engine
        engine=Engine.__new__(Engine);engine.torch=torch;engine.device='cpu';engine.dim=4
        engine.prefix=Mock(return_value='prompt');captured=[]
        class Inputs(dict):
            @property
            def input_ids(self):return self['input_ids']
            def to(self,device):return self
        class Tokenizer:
            pad_token_id=0
            def __call__(self,*a,**kw):return Inputs(input_ids=torch.ones((1,3),dtype=torch.long))
            def decode(self,*a,**kw):return 'reply'
        class Block:
            def register_forward_hook(self,hook):self.hook=hook;return Mock()
        class Model:
            dtype=torch.float32
            generation_config=type('Config',(),{'eos_token_id':2})()
            def generate(self,**kw):
                captured.append(engine.block.hook(None,None,torch.zeros((1,3,4))))
                captured.append(engine.block.hook(None,None,torch.zeros((1,1,4))))
                return torch.tensor([[1,1,1,2]])
        engine.tokenizer=Tokenizer();engine.block=Block();engine.model=Model()
        total=np.array([1.,2.,3.,4.]);full=np.array([0.,0.,3.,4.])
        engine.generate(['q'],[total],prefill_shifts=[full])
        np.testing.assert_allclose(captured[0][0,0],full)
        np.testing.assert_allclose(captured[0][0,-1],total)
        np.testing.assert_allclose(captured[1][0,0],total)
        with self.assertRaises(ValueError):engine.generate(['q'],[total],prefill_shifts=[np.full(4,np.nan)])


if __name__=='__main__':unittest.main()
