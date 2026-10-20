"""Real local HTTP → frozen Qwen weights → mixture and Walker end-to-end."""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from studio import profile_for, PRESETS

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda");args=p.parse_args()
    out=ROOT/"studio_runs/character-v2";out.mkdir(exist_ok=True)
    with (out/"live_server.log").open("w") as log:
        server=subprocess.Popen([sys.executable,str(ROOT/"serve.py"),"--run",str(ROOT/"runs/job-2117172"),"--device",args.device,"--port","8769","--max-tokens","256"],stdout=log,stderr=log)
        def request(path,obj=None):
            body=json.dumps(obj,ensure_ascii=False).encode() if obj is not None else None
            req=urllib.request.Request("http://127.0.0.1:8769"+path,data=body,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=300) as r:return json.loads(r.read())
        try:
            deadline=time.monotonic()+120
            while True:
                if server.poll() is not None: raise RuntimeError("Local server stopped during startup")
                try:
                    status=request("/api/status");break
                except urllib.error.URLError:
                    if time.monotonic()>deadline:raise
                    time.sleep(1)
            assert status["online"]
            profile=profile_for(PRESETS[1]);profile["role"]=dict(name="青禾",identity="你是书店店主青禾，喜欢观察雨天的城市。保持第一人称角色口吻。",facts=["书店叫纸舟。","你的朋友叫小满。","店里有一只叫墨点的猫。"],boundaries="保持角色事实，避免重复和无关解释。")
            scene=dict(id="fresh_cat",question="墨点把书推到地上，客人笑了。你说一句什么，再做一个动作？")
            comparison=request("/api/studio/compare",dict(profile=profile,scene=scene))
            assert set(comparison["results"])=={"baseline","prompt","geometry"}
            assert comparison["profile"]["role"]["name"]=="青禾" and comparison["score"] is None
            intervention=comparison["intervention"]
            if intervention.get('regularization')=='direction_projection':
                assert max(map(abs,intervention['projections'].values()),default=0.)<=intervention['projection_limit']+1e-5
            elif intervention.get('regularization','norm')=='norm':
                assert intervention['norm']<=intervention['cap']+1e-5
            else:
                assert intervention['cap_factor']==1.
            assert all(r["text"] and r["new_tokens"]==len(r["token_ids"]) for r in comparison["results"].values())
            bad=profile_for(PRESETS[1]);bad["coefficients"]["courage"]=True
            try:
                request("/api/studio/compare",dict(profile=bad,scene=scene));raise AssertionError("Boolean strength accepted")
            except urllib.error.HTTPError as error:assert error.code==400
            scenes=[dict(id="fresh_rain",question="雨天有人走进书店，说想读一页让自己振作的文字。你会如何回应？"),dict(id="fresh_loss",question="小满说自己弄丢了心爱的书。你会如何回应，并提出一个小行动？")]
            walked=request("/api/studio/walk",dict(profile=profile,scenes=scenes))
            assert len(walked["rankings"])==5 and len(walked["records"])==10
            assert all(r["judge_raw"]["text"] for r in walked["records"])
            assert walked["heldout"] is False
            result=dict(checks=["fresh custom role through real local HTTP", "three real comparator generations", "bounded mixture", "invalid controls rejected before inference", "five Walker candidates × two fresh scenes", "real 4B judge, no claim of held-out improvement"],comparison=comparison,walker=walked,manual_votes=0)
            (out/"live_api_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2));print("LIVE STUDIO OK",walked["winner"]["id"] if walked["winner"] else "no eligible candidate",flush=True)
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()

if __name__=="__main__": main()
