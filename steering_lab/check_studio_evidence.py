"""Reconstruct every mixture and the frozen decision from saved raw records."""
import argparse
import json
from pathlib import Path
import numpy as np
from studio import mixed_shift, parse_ratings, choose_winner, digest, validate_profile

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,default=Path("studio_runs/character-v2"));args=p.parse_args()
    data=json.loads((args.run/"studio_evidence.json").read_text())
    manifest=data["manifest"];geo=np.load(Path("runs/job-2117172/geometry.npz"));cap=manifest["provenance"]["norm_cap"]
    for trial in data["trials"]:
        profile=validate_profile(trial["profile"])
        assert digest(profile)==trial["profile_sha256"]
        _,info=mixed_shift(profile,geo["directions"],cap)
        assert info==trial["intervention"]
        assert trial["provenance"]==manifest["provenance"]
        for mode,record in trial["results"].items():
            assert record["ratings"]==parse_ratings(record["judge_raw"]["text"])
            assert record["new_tokens"]==len(record["token_ids"])
            assert record["text"]
    walker=data["walker"];calibration=[r for r in walker["records"] if r["split"]=="search"]
    decision=choose_winner(calibration,walker["requested_profile"])
    assert decision["winner"]==walker["winner"] and decision["rankings"]==walker["rankings"]
    frozen=json.loads((args.run/"frozen_choice.json").read_text())
    assert frozen["winner"]==walker["winner"] and frozen["selection_manifest_sha256"]==digest(manifest)
    assert not set(manifest["search_scenes"]) & set(manifest["check_scenes"])
    for record in walker["records"]: assert record["ratings"]==parse_ratings(record["judge_raw"]["text"])
    result=dict(checks=["all real trial token counts", "all bounded mixtures reconstructed", "all judge ratings match raw model text", "frozen choice reproduced using search scenes only", "check scenes excluded from selection"],trials=len(data["trials"]),manual_votes=0)
    (args.run/"evidence_check.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=="__main__": main()
