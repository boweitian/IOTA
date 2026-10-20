import unittest
import numpy as np
from scipy.sparse import csr_matrix
from fit_asset_router import set_loss, valid_actions, fit, vectorizer


class AssetFitTests(unittest.TestCase):
    def test_entity_numbers_and_operators_are_retained(self):
        x = vectorizer().fit_transform(['Austria', 'Australia', '4 + 2', '4 - 2']).toarray()
        self.assertFalse(np.array_equal(x[0], x[1]))
        self.assertFalse(np.array_equal(x[2], x[3]))

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        x = csr_matrix(rng.normal(size=(4, 3)))
        valid = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
        parameters = rng.normal(size=12)
        _, analytic = set_loss(parameters, x, valid)
        numeric = np.empty_like(parameters)
        for i in range(len(parameters)):
            lo, hi = parameters.copy(), parameters.copy()
            lo[i] -= 1e-6; hi[i] += 1e-6
            numeric[i] = (set_loss(hi, x, valid)[0] - set_loss(lo, x, valid)[0]) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_uncovered_rows_are_not_labeled_as_solved(self):
        data = dict(suites=np.array(['gsm8k', 'bbh', 'truthfulqa']),
                    reward=np.array([[0, 0, 0], [0, 1, 1], [.2, .3, .25]]))
        valid, covered = valid_actions(data)
        np.testing.assert_equal(covered, [False, True, True])
        np.testing.assert_equal(valid, [[False, False, False], [False, True, True], [False, True, False]])

    def test_linear_training_fits_success_sets(self):
        x = csr_matrix(np.eye(4))
        valid = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
        w, bias = fit(x, valid, [])
        choices = (x @ w + bias).argmax(1)
        self.assertTrue(valid[np.arange(4), choices].all())


if __name__ == '__main__':
    unittest.main()
