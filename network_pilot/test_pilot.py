import json
import unittest
from pathlib import Path
import numpy as np
from space import bounded_shift,task_graph,action,singles,proposal_pool
from experiment import parse_rubric,score_ifeval,validation_finalists


class PilotTests(unittest.TestCase):
    def test_actual_shift_bound_in_nonorthogonal_dictionary(self):
        directions=np.array([[1.,0.],[1.,0.]])
        shift,w=bounded_shift([1.,1.],directions,1.)
        np.testing.assert_allclose(shift,[1.,0.]);np.testing.assert_allclose(w,[.5,.5])
        shift,w=bounded_shift([1.,-1.],directions,1.)
        np.testing.assert_allclose(shift,[0.,0.]);np.testing.assert_allclose(w,[1.,-1.])

    def test_noop_and_signed_nodes(self):
        bank=singles(np.eye(3),1.)
        self.assertEqual(len(bank),7);self.assertEqual(bank[0]["norm"],0)
        np.testing.assert_allclose(bank[1]["shift"],[-.5,0,0])
        np.testing.assert_allclose(bank[2]["shift"],[.5,0,0])

    def test_validation_keeps_noop_when_search_ranks_it_last(self):
        bank=singles(np.eye(2),1.)
        records={("x",a["id"]):dict(reward=0 if a["id"]=="none" else 1) for a in bank}
        selected=validation_finalists(["x"],records,bank,3)
        self.assertIn("none",[a["id"] for a in selected])
        self.assertEqual(len(selected),4)

    def test_pair_interaction_measures_joint_effect(self):
        records={}
        for i in ["x","y"]:
            for name,reward in [("none",.2),("node:0:+1",.3),("node:1:+1",.4),("pair:0:1",.7)]:
                records[i,name]=dict(reward=reward)
        edges,g=task_graph(["x","y"],records,2)
        self.assertAlmostEqual(edges[0]["interaction_mean"],.2)
        self.assertAlmostEqual(g[0,1],.2/3)
        records["x","pair:0:1"]["reward"]=.5;records["y","pair:0:1"]["reward"]=.5
        self.assertAlmostEqual(task_graph(["x","y"],records,2)[1][0,1],0)

    def test_bad_judge_stays_missing(self):
        self.assertIsNone(parse_rubric('{"scores":[7]}',2))
        self.assertIsNone(parse_rubric('{"scores":[true,7]}',2))
        self.assertIsNone(parse_rubric('{"scores":[NaN,7]}',2))
        self.assertEqual(parse_rubric('{"scores":[6,8]}',2)["reward"],.7)

    def test_proposals_are_reproducible_and_bounded(self):
        d=np.array([[1.,0.,0.],[.8,.6,0.],[0.,0.,1.]])
        a=proposal_pool(11,3,d,1.,count=12);b=proposal_pool(11,3,d,1.,count=12)
        self.assertEqual([x["id"] for x in a],[x["id"] for x in b])
        self.assertTrue(all(x["norm"]<=1.+1e-12 for x in a))

    def test_fresh_disjoint_inputs_and_ifeval_groups(self):
        root=Path(__file__).resolve().parent;protocol=json.loads((root/"inputs/protocol.json").read_text())
        old=json.loads((root.parent/"steering_lab/inputs/manifest.json").read_text())
        used={r["id"] for r in old["records"]};rows=protocol["records"]
        self.assertFalse(used&{r["id"] for r in rows});self.assertEqual(len(rows),len({r["id"] for r in rows}))
        sets=[{r["question"] for r in rows if r["split"]==s} for s in ["search","validation","test"]]
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])
        groups=[{r["group"] for r in rows if r["suite"]=="ifeval" and r["split"]==s} for s in ["search","validation","test"]]
        for i in range(3):
            for j in range(i):self.assertFalse(groups[i]&groups[j])

    def test_official_checker_distinguishes_followed_constraints(self):
        row=dict(id="fixture",source_index=0,question="Include the word orbit twice.",
            instruction_id_list=["keywords:frequency"],kwargs=[dict(keyword="orbit",frequency=2,relation="at least")])
        self.assertEqual(score_ifeval(row,"orbit orbit")["reward"],1)
        self.assertEqual(score_ifeval(row,"orbit")["reward"],0)


if __name__=="__main__":unittest.main()
