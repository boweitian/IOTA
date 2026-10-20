"""Behavioral checks for evidence isolation, adaptive refinement and provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from iota.backends import SyntheticBackend, ReplayBackend, balanced_readout
from iota.discovery import (DiscoveryWalker, EvidenceMap, Rule, confirm_map,
                            compare_confirmations, digest, make_scenarios,
                            prediction_metrics, soulprint)


class DiscoveryTests(unittest.TestCase):
    def test_budget_charges_shared_anchors_and_stops_before_overrun(self):
        backend = SyntheticBackend()
        backend.cost_per_query = 2
        m = DiscoveryWalker(backend).run(budget=5)
        self.assertEqual(backend.physical_queries,2)
        self.assertEqual(m.version,2)
        self.assertEqual(sum(o['query_cost'] for o in m.observations),4)
        self.assertEqual(m.trajectory[-1]['budget_remaining'],1)

    def test_active_selection_changes_when_evidence_changes(self):
        walker = DiscoveryWalker(SyntheticBackend(),seed=2)
        m = walker.run(24)
        self.assertTrue(any(s['proposed_rules'] for s in m.trajectory))
        self.assertTrue(any(s['counterexample_to'] for s in m.trajectory))
        self.assertTrue(any(s['best_before'] != s['best_after'] for s in m.trajectory))
        self.assertTrue(all(o['split']=='explore' for o in m.observations))
        for step in m.trajectory[3:]:
            self.assertEqual(step['selection_reason'],'maximum expected hypothesis information gain')

    def test_confirmation_does_not_update_hypotheses(self):
        backend = SyntheticBackend()
        m = DiscoveryWalker(backend,seed=2).run(24)
        frozen = digest(m.to_dict())
        confirmed = confirm_map(m,backend)
        self.assertEqual(frozen,digest(m.to_dict()))
        self.assertEqual(confirmed['map_hash'],frozen)
        identity = soulprint(m,confirmed)
        self.assertEqual(identity['global_invariant_markers'],[])
        self.assertIsNone(identity['internal_geometry'])
        self.assertTrue(all(o['split']=='confirm' for o in confirmed['observations']))
        with self.assertRaises(ValueError):
            confirm_map(m,backend,make_scenarios('explore'))
        with self.assertRaises(ValueError):
            confirm_map(m,SyntheticBackend('claim_trusting'))

    def test_roundtrip_preserves_predictions_and_evidence(self):
        m = DiscoveryWalker(SyntheticBackend(),seed=2).run(24)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'map.json'
            m.save(path)
            loaded = EvidenceMap.load(path)
        self.assertEqual(digest(m.to_dict()),digest(loaded.to_dict()))
        self.assertEqual([m.predict(s.facts) for s in make_scenarios('confirm')],
                         [loaded.predict(s.facts) for s in make_scenarios('confirm')])
        for obs in m.observations:
            raw = {k:v for k,v in obs.items() if k!='evidence_id'}
            self.assertEqual(obs['evidence_id'],digest(raw))

    def test_cross_model_comparison_matches_scenarios_not_coordinates(self):
        a_backend,b_backend = SyntheticBackend(),SyntheticBackend('claim_trusting')
        a = confirm_map(DiscoveryWalker(a_backend).run(8),a_backend)
        b = confirm_map(DiscoveryWalker(b_backend).run(8),b_backend)
        result = compare_confirmations(a,b)
        self.assertGreater(result['decision_disagreement'],0)
        self.assertEqual(result['quality_status'],'descriptive_only')
        sensitive = copy.deepcopy(a)
        sensitive['metrics']['mapping_disagreement_fraction']=1.
        self.assertEqual(compare_confirmations(sensitive,b)['quality_status'],'unreliable_readout_comparison')
        invalid = copy.deepcopy(b)
        invalid['observations'][0]['prompt_hash'] = 'tampered'
        with self.assertRaises(ValueError):
            compare_confirmations(a,invalid)
        changed = copy.deepcopy(a)
        changed['map_hash']='wrong'
        with self.assertRaises(ValueError):
            soulprint(DiscoveryWalker(a_backend).run(8),changed)

    def test_known_counterexample_refines_consent_into_scope_condition(self):
        m = EvidenceMap({'kind':'synthetic'},0,'active',rules=[Rule('grant',(('consent',True),))])
        facts = dict(public=False,consent=True,verified=True,scope=False,revoked=False,claim=False)
        m.observations.append({'facts':facts,'disclose':False})
        proposals = m.propose_refinements('counterexample')
        self.assertTrue(proposals)
        self.assertTrue(any(('scope',True) in r.literals for r in m.rules))

    def test_class_imbalance_is_visible(self):
        labels = [{'disclose':False}]*31+[{'disclose':True}]
        metrics = prediction_metrics([.1]*32,labels)
        self.assertEqual(metrics['accuracy_to_model'],31/32)
        self.assertEqual(metrics['balanced_accuracy_to_model'],.5)
        self.assertEqual(metrics['disclose_recall'],0)

    def test_seed_reproducibility(self):
        a = DiscoveryWalker(SyntheticBackend(),seed=4).run(12)
        b = DiscoveryWalker(SyntheticBackend(),seed=4).run(12)
        self.assertEqual(digest(a.to_dict()),digest(b.to_dict()))

    def test_resume_preserves_future_selection(self):
        full = DiscoveryWalker(SyntheticBackend(),seed=2).run(24)
        part = DiscoveryWalker(SyntheticBackend(),seed=2).run(9)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'map.json'
            part.save(path)
            loaded = EvidenceMap.load(path)
        resumed = DiscoveryWalker.resume(SyntheticBackend(),loaded).run(15)
        self.assertEqual([o['scenario_id'] for o in full.observations],
                         [o['scenario_id'] for o in resumed.observations])
        self.assertEqual(full.best_rule()[0],resumed.best_rule()[0])

    def test_corrupt_evidence_is_rejected_on_load(self):
        m = DiscoveryWalker(SyntheticBackend()).run(4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'map.json'
            m.save(path)
            data = json.loads(path.read_text())
            data['observations'][0]['p_disclose']=.123
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                EvidenceMap.load(path)

    def test_saved_measurements_replay_without_a_model(self):
        m = DiscoveryWalker(SyntheticBackend(),seed=2).run(24)
        backend = ReplayBackend(m)
        replayed = DiscoveryWalker(backend,'active',2).run(24)
        self.assertEqual(digest(m.to_dict()),digest(replayed.to_dict()))
        self.assertEqual(backend.physical_queries,0)

    def test_constant_readout_is_not_certified_identity(self):
        backend = SyntheticBackend()
        backend.query = lambda s: dict(disclose=True,p_disclose=.9,selected_label='DISCLOSE',measurement_kind='synthetic_fixture')
        m = DiscoveryWalker(backend).run(4)
        confirmed = confirm_map(m,backend)
        self.assertTrue(confirmed['metrics']['constant_decision_readout'])
        self.assertIsNone(confirmed['metrics']['balanced_accuracy_to_model'])
        self.assertEqual(soulprint(m,confirmed)['status'],'insufficient_readout_diversity')

    def test_balanced_readout_cancels_additive_label_bias(self):
        # Semantic disclosure advantage = 2; A-label bias = 5.
        p,orders = balanced_readout([7,0,5,2])
        unbiased,_ = balanced_readout([2,0,0,2])
        self.assertAlmostEqual(p,unbiased)
        self.assertTrue((orders[0]>=.5)!=(orders[1]>=.5))
        p,_ = balanced_readout([1000,0,1000,0])
        self.assertEqual(p,.5)

    def test_mapping_sensitive_readout_does_not_become_semantic_relation(self):
        backend = SyntheticBackend()
        original = backend.query
        backend.query = lambda s: dict(original(s),mapping_decision_disagreement=True)
        m = DiscoveryWalker(backend).run(6)
        confirmed = confirm_map(m,backend)
        identity = soulprint(m,confirmed)
        self.assertEqual(identity['status'],'mapping_sensitive_readout')
        self.assertTrue(all(r['status']=='unreliable_readout_effect' for r in identity['relations']))

    def test_persisted_hash_survives_numerical_summary_rounding(self):
        import math
        m = DiscoveryWalker(SyntheticBackend(),seed=2).run(8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'map.json'
            data = m.to_dict()
            data['hypotheses'][0]['conditional_weight'] = math.nextafter(data['hypotheses'][0]['conditional_weight'],1)
            path.write_text(json.dumps(data))
            loaded = EvidenceMap.load(path)
            self.assertEqual(loaded.content_hash,digest(data))
            loaded.save(path)
            self.assertEqual(digest(json.loads(path.read_text())),digest(data))


if __name__ == '__main__':
    unittest.main()
