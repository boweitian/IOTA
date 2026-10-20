"""Control bounds and selection leakage are product-critical invariants."""
import unittest
import numpy as np
from studio import AXES, PRESETS, SCENES, CharacterStudio, profile_for, validate_profile, mixed_shift, candidates, choose_winner, parse_ratings, retention_check

class StudioTests(unittest.TestCase):
    def test_correlated_mixture_is_capped_and_signed(self):
        profile=profile_for(PRESETS[1]);directions=np.ones((len(AXES),4))
        delta,info=mixed_shift(profile,directions,1)
        self.assertAlmostEqual(np.linalg.norm(delta),1)
        self.assertLess(info["cap_factor"],1)
        profile["coefficients"]={k:-v for k,v in profile["coefficients"].items()}
        reverse,_=mixed_shift(profile,directions,1)
        np.testing.assert_allclose(reverse,-delta)

    def test_recipe_rejects_nonfinite_or_unknown_controls(self):
        for value in [float("nan"),float("inf"),True,2.1]:
            profile=profile_for(PRESETS[1]);profile["coefficients"]["courage"]=value
            with self.assertRaises(ValueError): validate_profile(profile)
        profile=profile_for(PRESETS[1]);profile["coefficients"]["unknown"]=.2
        with self.assertRaises(ValueError): validate_profile(profile)

    def test_user_defined_axis_is_bound_to_its_instructions(self):
        from studio import custom_axis_id
        axis=dict(name="幽默",low="严肃",high="幽默",positive="用轻松幽默的语气回应。",negative="严肃地回应。")
        profile=profile_for(PRESETS[1]);profile["custom_axes"]=[axis];cid=custom_axis_id(axis["positive"],axis["negative"])
        profile["coefficients"][cid]=.5
        clean=validate_profile(profile)
        self.assertEqual(clean["custom_axes"][0]["id"],cid);self.assertEqual(clean["coefficients"][cid],.5)
        directions=np.zeros((len(AXES),4));delta,_=mixed_shift(clean,directions,10,{cid:np.array([2.,0,0,0])})
        np.testing.assert_allclose(delta,[1.,0,0,0])
        bad=profile_for(PRESETS[1]);bad["custom_axes"]=[dict(axis,id="custom-000000000000")]
        with self.assertRaises(ValueError): validate_profile(bad)
        orphan=profile_for(PRESETS[1]);orphan["coefficients"][cid]=.5
        with self.assertRaises(ValueError): validate_profile(orphan)
        same=profile_for(PRESETS[1]);same["custom_axes"]=[dict(axis,negative=axis["positive"])]
        with self.assertRaises(ValueError): validate_profile(same)

    def test_check_scenes_never_select_recipe(self):
        profile=profile_for(PRESETS[1]);rows=[]
        for candidate in candidates(profile):
            for scene in SCENES:
                ratings={k:7 for k in ["identity","coherence","imagery","warmth","courage","brevity"]}
                if candidate["id"]=="neutral" and scene["split"]=="search": ratings={k:8 for k in ratings}
                if candidate["id"]=="current" and scene["split"]=="check": ratings={k:10 for k in ratings}
                rows.append(dict(candidate=candidate["id"],scene=scene,split=scene["split"],ratings=ratings,truncated=False))
        self.assertEqual(choose_winner(rows,profile)["winner"]["id"],"neutral")
        for row in rows:
            if row["candidate"]=="neutral" and row["split"]=="search": row["truncated"]=True
        self.assertNotEqual(choose_winner(rows,profile)["winner"]["id"],"neutral")

    def test_one_scene_is_insufficient_and_missing_judge_is_unknown(self):
        profile=profile_for(PRESETS[1]);row=dict(candidate="current",scene=SCENES[0],split="search",ratings={k:9 for k in ["identity","coherence","imagery","warmth","courage","brevity"]},truncated=False)
        self.assertIsNone(choose_winner([row],profile)["winner"])
        self.assertIsNone(parse_ratings('{"identity":9}'))
        self.assertEqual(retention_check(SCENES[0],"")["status"],"not_applicable")

    def test_duplicate_scene_identifiers_fail_before_inference(self):
        studio=CharacterStudio.__new__(CharacterStudio)
        with self.assertRaises(ValueError):
            studio.walk(profile_for(PRESETS[1]),[dict(id="same",question="One?"),dict(id="same",question="Two?")])

if __name__=="__main__": unittest.main()
