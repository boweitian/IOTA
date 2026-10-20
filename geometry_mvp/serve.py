#!/usr/bin/env python3
"""Serve the offline atlas, optionally with a real local activation-steering API."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--live",action="store_true",help="Load the pinned model for custom prompts")
    p.add_argument("--device",choices=["cuda","cpu"],default="cpu")
    p.add_argument("--data",type=Path,default=ROOT/"data")
    args=p.parse_args()
    atlas=json.loads((args.data/"atlas.json").read_text())
    expected_hash=atlas["directions_sha256"]
    if hashlib.sha256((args.data/"directions.npz").read_bytes()).hexdigest()!=expected_hash:
        raise SystemExit("Measured direction artifact hash mismatch")
    engine=None;directions=None;lock=threading.Lock()
    if args.live:
        from engine import SteeringEngine
        engine=SteeringEngine(args.device,atlas["model"]["layer_index"])
        for key in ("name","revision","layer_index","scale","system_prompt","chat_template_sha256"):
            if engine.metadata[key]!=atlas["model"][key]:
                raise SystemExit(f"Live model differs from measurement: {key}")
        directions=engine.load_directions(args.data/"directions.npz")
    page=(ROOT/"dist/index.html").read_bytes()
    # An alternate data directory must also be built into the page.
    if expected_hash.encode() not in page:
        raise SystemExit("Build the page with this data directory first")
    class Handler(BaseHTTPRequestHandler):
        def send_body(self,status,body,content_type="application/json; charset=utf-8"):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            try:
                self.send_response(status);self.send_header("Content-Type",content_type)
                self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
                self.end_headers();self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):
                # Closing the tab during generation is not a model failure.
                return
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ("/","/index.html"):
                self.send_body(200,page,"text/html; charset=utf-8")
            elif path=="/api/status":
                self.send_body(200,dict(live=bool(engine),model=atlas["model"]["name"],directions_sha256=expected_hash))
            elif path=="/favicon.ico":
                self.send_body(204,b"","image/x-icon")
            else:self.send_body(404,dict(error="Not found"))
        def do_POST(self):
            if urlsplit(self.path).path!="/api/compare":
                return self.send_body(404,dict(error="Not found"))
            if engine is None:
                return self.send_body(503,dict(error="实时模型未连接；启动服务时添加 --live。"))
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=16384:raise ValueError("Request too large or empty")
                body=json.loads(self.rfile.read(length))
                prompt=body["prompt"];axis=body["axis"];strength=body["strength"]
                base_strength=body.get("baseline_strength",0)
                if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=1200:
                    raise ValueError("问题需包含 1–1200 个字符。")
                if axis not in directions or isinstance(strength,bool) or strength not in atlas["strengths"]:
                    raise ValueError("Unknown direction or strength")
                if isinstance(base_strength,bool) or base_strength not in atlas["strengths"]:
                    raise ValueError("Invalid baseline strength")
                # Reject off-origin form submissions to this loopback inference service.
                origin=self.headers.get("Origin")
                if origin and urlsplit(origin).netloc!=self.headers.get("Host"):
                    raise ValueError("Cross-origin requests are not supported")
            except (ValueError,KeyError,TypeError) as exc:
                return self.send_body(400,dict(error=str(exc)))
            if not lock.acquire(blocking=False):
                return self.send_body(409,dict(error="模型正在生成，请等待当前请求完成。"))
            try:
                if strength==base_strength:
                    baseline=engine.generate_batch(prompt,[base_strength*directions[axis]])[0];steered=dict(baseline)
                else:baseline,steered=engine.generate_batch(prompt,[base_strength*directions[axis],strength*directions[axis]])
                self.send_body(200,dict(baseline=baseline,steered=steered,axis=axis,strength=strength,
                                       baseline_strength=base_strength,
                                       mode="activation_steering",weights_modified=False))
            except Exception as exc:
                print("Generation error:",repr(exc),flush=True)
                self.send_body(500,dict(error="模型生成失败，详情见服务日志。"))
            finally:lock.release()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"IOTA: http://127.0.0.1:{args.port} · {'live model' if engine else 'offline replay'}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
