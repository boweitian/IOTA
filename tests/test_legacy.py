"""Checks that the legacy activation path cannot emit fabricated identity fields."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Optional ML dependencies required for legacy checks')
class LegacyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from iota.atlas import AtlasBuilder,ConceptDirection,SemanticMap
        self.np,self.Direction,self.Map = np,ConceptDirection,SemanticMap
        class FakeBuilder(AtlasBuilder):
            @property
            def num_layers(self):
                return 6
            def _extract_direction(self,a,b,positions=None,layer=None):
                self.last_requested_layer = layer
                return np.array([1.,2.]),np.zeros(2),0,0
        self.builder = FakeBuilder('fake',device='cpu')

    def test_measured_layer_is_forwarded_and_recorded(self):
        _,metrics = self.builder._extract_direction_multi_context('a','b',['{W} one','{W} two'],layer=3)
        self.assertEqual(self.builder.last_requested_layer,3)
        self.assertEqual(metrics['hidden_state_index'],3)
        _,metrics = self.builder._extract_direction_multi_context('a','b',['{W} one','{W} two'])
        self.assertEqual(metrics['hidden_state_index'],6)

    def test_single_context_does_not_produce_nan_loo(self):
        with self.assertRaises(ValueError):
            self.builder._extract_direction_multi_context('a','b',['{W}'])

    def test_peak_layer_requires_measurement(self):
        with self.assertRaises(NotImplementedError):
            self.builder._find_peak_layer(('a','b'),['{W}'])

    def test_unmeasured_soul_fields_remain_empty(self):
        from iota.soulprint import generate_soulprint
        d = self.Direction('a','b',self.np.array([1.,0.]),['{W}'],6,
                           {'loo_cosine_mean':.95,'loo_cosine_std':.01,'n_contexts':40})
        m = self.Map('fake','synthetic-test',{'a|b':d},{})
        identity = generate_soulprint(m)
        self.assertEqual(identity.invariant_markers,[])
        self.assertEqual(identity.depth_profiles,{})
        self.assertEqual(identity.behavioral_tendencies,{})
        self.assertIsNone(identity.candidate_markers[0].confidence)

    def test_cross_model_raw_vector_comparison_is_rejected(self):
        from iota.atlas import compare_maps
        a = self.Direction('a','b',self.np.array([1.,0.]),[],1)
        b = self.Direction('a','b',self.np.array([0.,1.]),[],1)
        with self.assertRaises(ValueError):
            compare_maps(self.Map('model-a','test',{'a|b':a},{}),
                         self.Map('model-b','test',{'a|b':b},{}))


if __name__ == '__main__':
    unittest.main()
