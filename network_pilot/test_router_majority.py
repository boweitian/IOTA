import unittest
import numpy as np
from router_majority import majority_targets


class MajorityTests(unittest.TestCase):
    def test_majority_never_overrides_sample_correctness(self):
        data=dict(suites=np.array(['gsm8k']*4),
             reward=np.array([[0,1,1],[0,1,0],[0,1,0],[1,0,0]],dtype=float))
        target,valid,covered,support=majority_targets(data)
        self.assertEqual(target.argmax(1).tolist(),[1,1,1,0])
        self.assertFalse((target & ~valid).any())
        self.assertEqual(support['gsm8k'].tolist(),[1,3,1])

    def test_votes_are_benchmark_specific_and_uncovered_has_no_label(self):
        data=dict(suites=np.array(['gsm8k','bbh','bbh','bbh','gsm8k']),
             reward=np.array([[1,1,0],[0,1,1],[0,0,1],[0,0,1],[0,0,0]],dtype=float))
        target,_,covered,_=majority_targets(data)
        self.assertEqual(target[0].argmax(),0)
        self.assertEqual(target[1].argmax(),2)
        self.assertFalse(covered[-1])
        self.assertFalse(target[-1].any())

    def test_continuous_rewards_preserve_sample_optimum(self):
        data=dict(suites=np.array(['truthfulqa']*3),
             reward=np.array([[.9,.8,.1],[.3,.4,.2],[.3,.4,.1]]))
        target,_,_,_=majority_targets(data)
        self.assertEqual(target.argmax(1).tolist(),[0,1,1])


if __name__=='__main__':
    unittest.main()
