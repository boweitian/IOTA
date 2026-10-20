"""Exercise the same runtime used by HTTP with fresh, non-benchmark tasks."""
import argparse
import json
from pathlib import Path
import numpy as np
from serve import LiveRouter
from router import predict

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path)
    args=p.parse_args();run=args.run.resolve()
    live=LiveRouter(run,max_tokens=192)
    questions=[("creative","Write a 100-word letter from a lighthouse keeper to his mother. Convey loneliness through specific objects and sounds rather than naming the emotion."),
               ("objective","A garden has 6 rows with 7 plants in each row. Three plants are removed. How many remain?")]
    records=[]
    for domain,question in questions:
        before=len(live.engine.block._forward_hooks)
        result=live.route(question,domain)
        assert len(live.engine.block._forward_hooks)==before
        fs=np.asarray([result["features"]])
        expected=predict(live.policies[domain],fs)[0][0]
        assert result["action"]==live.actions[int(expected)]["id"]
        assert result["baseline"]["token_ids"] and result["steered"]["token_ids"]
        assert len(result["features"])==18 and result["score"] is None
        records.append(result)
        print(domain,result["action"],"hook cleanup OK",flush=True)
    output=dict(checks=["new tasks use the frozen geometry policy without targets",
        "native activation features and selected action agree",
        "baseline and steered responses generated with real token IDs",
        "intervention hooks removed after both requests"],records=records)
    (run/"live_check.json").write_text(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
