import json
import unittest
from pathlib import Path
import numpy as np
from grading import grade, extract
from judge import parse_score, parse_winner
from router import geometry_features, fit, predict

class ProtocolTests(unittest.TestCase):
    def test_disjoint_prompt_splits(self):
        root=Path(__file__).resolve().parent
        for file in ["manifest.json","art_manifest.json"]:
            rows=json.loads((root/"inputs"/file).read_text())["records"]
            dev={r["id"] for r in rows if r["split"]=="dev"}
            test={r["id"] for r in rows if r["split"]=="test"}
            self.assertFalse(dev & test)
            self.assertEqual(len(dev)+len(test),len(rows))
            self.assertFalse({r["question"] for r in rows if r["split"]=="dev"} & {r["question"] for r in rows if r["split"]=="test"})

    def test_numeric_grade_does_not_guess_from_reasoning(self):
        row=dict(task="gsm8k",target="1200")
        self.assertTrue(grade("I multiply 12 by 100.\nFINAL: $1,200",row)["correct"])
        self.assertIsNone(extract("FINAL: 12 or 100", "gsm8k"))
        self.assertFalse(grade("FINAL: 12",row)["correct"])
        self.assertEqual(extract("FINAL: False","boolean_expressions"),"false")
        self.assertEqual(extract("FINAL: (C)","logical_deduction_five_objects"),"C")

    def test_policy_has_real_noop_and_can_choose_by_context(self):
        x=np.array([[-1.],[-1.],[1.],[1.]])
        values=np.array([[0,1],[0,1],[1,0],[1,0]],dtype=float)
        p=fit(x,values,[0,1],1,0)
        choices,_=predict(p,np.array([[-1.],[1.]]))
        self.assertEqual(choices.tolist(),[1,0])
        p=fit(x,np.ones((4,2)),[0,1],1,0)
        self.assertEqual(predict(p,x)[0].tolist(),[0]*4)

    def test_geometry_features_are_native_direction_projections(self):
        h=np.array([[3.,4.]])
        d=np.array([[1.,0.],[0.,1.]])
        np.testing.assert_allclose(geometry_features(h,h,d),[[.6,.8,.6,.8]])

    def test_bad_judge_output_remains_missing(self):
        self.assertIsNone(parse_score('{"faithfulness": 12}'))
        self.assertIsNone(parse_score('excellent'))
        self.assertIsNone(parse_winner('both are good'))
        score=parse_score('{"faithfulness": 8,"coherence": 7,"originality": 6,"craft": 9}')
        self.assertEqual(score["score"],.75)

if __name__=="__main__":
    unittest.main()
