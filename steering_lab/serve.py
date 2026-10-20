"""Loopback-only local demo; the frozen router never sees benchmark targets."""
import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from concepts import task_prompt
from engine import Engine
from experiment import actions
from router import geometry_features,predict
from studio import CharacterStudio

ROOT=Path(__file__).resolve().parent

class LiveRouter:
    def __init__(self,run,device="cuda",max_tokens=512):
        self.protocol=json.loads((run/"protocol.json").read_text())
        self.engine=Engine(self.protocol["objective"],device=device)
        geometry=dict(np.load(run/"geometry.npz"))
        self.unit=geometry["unit"]
        self.actions=actions(geometry)
        self.policies={kind:json.loads((run/f"{kind}_policy.json").read_text())["geometry"] for kind in ["creative","objective"]}
        self.lock=threading.Lock();self.max_tokens=max_tokens

    def route(self,question,domain):
        if domain not in self.policies or not isinstance(question,str) or not question.strip() or len(question)>5000:
            raise ValueError("Provide a question of 1–5000 characters and choose creative or objective")
        with self.lock:
            q=question.strip() if domain=="creative" else task_prompt(question.strip())
            pooled,last=self.engine.measure([q])
            fs=geometry_features(pooled,last,self.unit)
            ix,traces=predict(self.policies[domain],fs)
            a=self.actions[int(ix[0])]
            generated=self.engine.generate([q,q],[self.actions[0]["shift"],a["shift"]],
                max_new_tokens=self.max_tokens if domain=="creative" else min(256,self.max_tokens))
            return dict(question=question,domain=domain,action=a["id"],trace=traces[0],features=fs[0].tolist(),
                baseline=generated[0],steered=generated[1],model=self.engine.meta,
                protocol_sha256=self.protocol["sha256"],score=None,
                note="New local inference, no gold answer or benchmark score used during routing")

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path)
    p.add_argument("--port",type=int,default=8768);p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--max-tokens",type=int,default=512);p.add_argument("--offline",action="store_true");args=p.parse_args()
    run=args.run.resolve()
    if not (ROOT/"dist/index.html").exists():
        raise RuntimeError("Build the completed report first")
    live=None if args.offline or not (run/"objective_policy.json").exists() else LiveRouter(run,args.device,args.max_tokens)
    studio=None if args.offline else CharacterStudio(run,args.device,min(args.max_tokens,512),
        engine=live.engine if live else None,lock=live.lock if live else None)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):
            super().__init__(*a,directory=str(ROOT/"dist"),**kw)
        def do_POST(self):
            if self.path not in ["/api/route","/api/studio/compare","/api/studio/walk","/api/studio/direction","/api/studio/chat"]:
                self.send_error(404);return
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0 < n <= 120000:
                    raise ValueError("Invalid request size")
                request=json.loads(self.rfile.read(n))
                if studio is None:
                    self.send_json(503,dict(error="Server started in offline mode"));return
                if not isinstance(request,dict):
                    raise ValueError("Request must be an object")
                if self.path=="/api/route":
                    if live is None:raise ValueError("This model session has no benchmark router")
                    result=live.route(request.get("question"),request.get("domain"))
                elif self.path=="/api/studio/chat":
                    result=studio.chat(request.get("profile"),request.get("messages"),request.get("binding"))
                elif self.path=="/api/studio/compare":
                    result=studio.compare(request.get("profile"),request.get("scene"))
                elif self.path=="/api/studio/direction":
                    result=studio.direction(request.get("axis"))
                else:
                    result=studio.walk(request.get("profile"),request.get("scenes"))
                self.send_json(200,result)
            except (ValueError,TypeError,json.JSONDecodeError) as e:
                self.send_json(400,dict(error=str(e)))
            except Exception as e:
                self.send_json(500,dict(error=str(e)))
        def send_json(self,status,obj):
            body=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        def end_headers(self):
            path=self.path.split('?',1)[0]
            if path=='/' or path.endswith(('.html','.js','.css')):
                self.send_header('Cache-Control','no-store, max-age=0')
            super().end_headers()
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/' or path.endswith(('.html','.js','.css')):
                if 'If-Modified-Since' in self.headers:del self.headers['If-Modified-Since']
            if self.path=="/api/status":
                self.send_json(200,dict(online=studio is not None,studio=studio.metadata() if studio else None));return
            super().do_GET()
    print(f"IOTA demo http://127.0.0.1:{args.port} | {'offline' if args.offline else args.device}",flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()

if __name__=="__main__":
    main()
