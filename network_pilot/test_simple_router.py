"""Checks for outcome leakage, multi-label routing and grouped evaluation."""
import unittest
import numpy as np
from simple_router import (SUITES, fit_model, folds, predict_spec, prompt_group,
                           rank_actions, scores)


class RouterTests(unittest.TestCase):
    def test_knn_keeps_multiple_successful_actions(self):
        x = np.eye(3)
        y = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=float)
        model = fit_model(x, y, dict(algorithm='knn', parameter=1))
        chosen = rank_actions(scores(model, x))[:, 0]
        np.testing.assert_equal(y[np.arange(3), chosen], y.max(1))
        # A successful no-op wins ties without peeking at inference labels.
        self.assertEqual(chosen[1], 0)

    def test_heldout_outcomes_do_not_affect_predictions(self):
        rng = np.random.default_rng(12)
        suites = np.repeat(SUITES, 4)
        data = dict(suites=suites, geometry=rng.normal(size=(20, 54)),
                    reward=rng.integers(0, 2, size=(20, 4)).astype(float),
                    texts=np.array([f'{s} question number {i}' for i, s in enumerate(suites)]),
                    actions=np.array(['none', 'a', 'b', 'c']))
        test = np.arange(3, 20, 4)
        train = np.setdiff1d(np.arange(20), test)
        for feature in ['text', 'geometry']:
            for algorithm in ['knn', 'ridge']:
                spec = dict(feature=feature, algorithm=algorithm, parameter=1)
                before = predict_spec(spec, data, train, test)
                changed = dict(data, reward=data['reward'].copy())
                changed['reward'][test] = 1000
                after = predict_spec(spec, changed, train, test)
                np.testing.assert_equal(before, after)

    def test_duplicate_prompts_never_cross_folds(self):
        suites = np.repeat(SUITES, 12)
        groups = np.array([prompt_group(f'{s} item {i//2}')
                           for s in SUITES for i in range(12)])
        data = dict(suites=suites, groups=groups)
        seen = []
        for train, test in folds(np.arange(60), data, 3, 17):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(60)))


if __name__ == '__main__':
    unittest.main()
