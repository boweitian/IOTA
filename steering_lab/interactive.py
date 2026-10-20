"""Start a persistent browser portal and an owned GPU session; no per-task shell commands."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime,timezone,timedelta
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parent;RUNTIME=ROOT/"runtime";ACTIVE=RUNTIME/"active.json"

def save(path,obj):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False));temp.replace(path)

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def portal_status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status",timeout=5) as r:
            result=json.loads(r.read())
            return result if result.get("application")=="iota-interactive" else None
    except (OSError,ValueError):return None

def scheduler_state(job_id):
    result=subprocess.run(["squeue","--noheader","--jobs",str(job_id),"--format","%T"],capture_output=True,text=True,timeout=3)
    if result.returncode==0:return result.stdout.strip() or "FINISHED"
    # Slurm removes completed jobs from squeue after a short retention period.
    # Do not keep the last RUNNING state when an expired job is no longer known.
    if "Invalid job id specified" in result.stderr:return "FINISHED"
    return None

def serve(session,port):
    meta=read(session/"session.json");cache=dict(checked=0,state="PENDING");lock=threading.Lock()
    def state():
        worker=read(session/"worker.json",dict(online=False,state="queued"))
        with lock:
            if time.monotonic()-cache["checked"]>10:
                try:
                    current=scheduler_state(meta["job_id"])
                    if current is not None:cache["state"]=current
                except (OSError,subprocess.TimeoutExpired):pass
                cache["checked"]=time.monotonic()
        online=worker.get("online",False) and time.time()-worker.get("updated_at",0)<600 and cache["state"] not in ["FINISHED","FAILED","CANCELLED","TIMEOUT"]
        message="模型已连接，可以直接提交新场景。" if online else "GPU 正在排队或加载，页面回放可以继续使用。"
        if cache["state"] in ["FINISHED","FAILED","CANCELLED","TIMEOUT"]:message="本次 GPU 会话已结束；保留实测回放，请在聊天中让我重新开启。"
        return dict(application="iota-interactive",online=bool(online),state=worker.get("state"),job_state=cache["state"],
            message=message,studio=worker.get("studio"),session_id=meta["session_id"],expires_at=meta["expires_at"])
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/"dist"),**kw)
        def send_json(self,status,obj):
            body=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Cache-Control","no-store")
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
            if self.path=="/api/status":self.send_json(200,state());return
            if self.path=="/api/latest":self.send_json(200,read(session/"latest.json",dict(latest=None)));return
            super().do_GET()
        def do_POST(self):
            if self.path not in ["/api/studio/compare","/api/studio/walk","/api/studio/direction","/api/studio/chat","/api/route"]:self.send_error(404);return
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0<n<=120000:raise ValueError("Invalid request size")
                body=json.loads(self.rfile.read(n))
                if not isinstance(body,dict):raise ValueError("Request must be an object")
                status=state()
                if not status["online"]:self.send_json(503,dict(error=status["message"]));return
                key=uuid.uuid4().hex;request=session/"requests"/(key+".json");response=session/"responses"/(key+".json")
                save(request,dict(path=self.path,body=body,submitted_at=time.time()))
                deadline=time.monotonic()+300
                while time.monotonic()<deadline:
                    result=read(response)
                    if result is not None:
                        if result["status"]==200 and body.get("display_in_browser") is True and self.path=="/api/studio/compare":
                            save(session/"latest.json",dict(id=key,completed_at=time.time(),result=result["body"]))
                        self.send_json(result["status"],result["body"]);response.unlink(missing_ok=True);return
                    time.sleep(.2)
                request.unlink(missing_ok=True);self.send_json(504,dict(error="本次生成超时，请稍后重试。"))
            except (ValueError,TypeError) as error:self.send_json(400,dict(error=str(error)))
            except (BrokenPipeError,ConnectionResetError):pass
    print(f"IOTA Interactive http://127.0.0.1:{port}",flush=True)
    ThreadingHTTPServer(("127.0.0.1",port),Handler).serve_forever()

def start(port,hours=4):
    RUNTIME.mkdir(exist_ok=True)
    existing=portal_status(port)
    if existing and existing.get("job_state") not in ["FINISHED","FAILED","CANCELLED","TIMEOUT"] and existing.get("state")!="stopped":
        print(json.dumps(existing,ensure_ascii=False,indent=2));return
    if existing:stop()
    session=RUNTIME/("session-"+uuid.uuid4().hex[:12]);session.mkdir(mode=0o700)
    (session/"requests").mkdir();(session/"responses").mkdir()
    result=subprocess.run(["sbatch","--parsable",f"--time={hours*60}",str(ROOT/"run_interactive.sbatch"),str(session)],capture_output=True,text=True,check=True)
    job_id=int(result.stdout.strip().split(";")[0]);now=datetime.now(timezone.utc)
    meta=dict(session_id=session.name,job_id=job_id,session=str(session),port=port,url=f"http://127.0.0.1:{port}",
        started_at=now.isoformat(),expires_at=(now+timedelta(hours=hours)).isoformat(),lease_hours=hours)
    save(session/"session.json",meta)
    with (session/"portal.log").open("a") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"interactive.py"),"serve","--session",str(session),"--port",str(port)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    meta["portal_pid"]=process.pid;save(ACTIVE,meta)
    for _ in range(20):
        if portal_status(port):print(json.dumps(meta,ensure_ascii=False,indent=2));return
        if process.poll() is not None:break
        time.sleep(.25)
    subprocess.run(["scancel",str(job_id)],check=False)
    raise RuntimeError("Browser portal could not bind its port; GPU job cancelled. See portal.log")

def stop():
    meta=read(ACTIVE)
    if not meta:print("No owned interactive session");return
    subprocess.run(["scancel",str(meta["job_id"])],check=False)
    pid=meta.get("portal_pid")
    try:
        command=Path(f"/proc/{pid}/cmdline").read_bytes()
        if str(ROOT/"interactive.py").encode() in command and str(meta["session"]).encode() in command:os.kill(pid,signal.SIGTERM)
    except (OSError,ProcessLookupError):pass
    ACTIVE.unlink(missing_ok=True);print("Stopped owned IOTA interactive session")

def refresh_portal(new_session=None):
    previous=read(ACTIVE)
    if not previous:raise RuntimeError("No owned interactive session")
    meta=previous
    if new_session is not None:
        new_session=Path(new_session).resolve()
        if new_session.parent!=RUNTIME.resolve():raise RuntimeError("Replacement session must be owned by this portal")
        meta=read(new_session/"session.json")
        worker=read(new_session/"worker.json",{})
        if not meta or not worker.get("online") or worker.get("state")!="ready" or time.time()-worker.get("updated_at",0)>60:
            raise RuntimeError("Replacement GPU worker is not ready")
        if scheduler_state(meta["job_id"])!="RUNNING":raise RuntimeError("Replacement GPU job is not running")
        if not meta.get("expires_at"):raise RuntimeError("Replacement GPU deadline has not been verified")
        meta=dict(meta,port=previous["port"],url=previous["url"])
    pid=previous["portal_pid"]
    try:
        command=Path(f"/proc/{pid}/cmdline").read_bytes()
        if str(ROOT/"interactive.py").encode() not in command or str(previous["session"]).encode() not in command:
            raise RuntimeError("Refusing to replace an unrelated process")
        os.kill(pid,signal.SIGTERM)
    except FileNotFoundError:pass
    time.sleep(.5)
    session=Path(meta["session"])
    with (session/"portal.log").open("a") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"interactive.py"),"serve","--session",str(session),"--port",str(meta["port"])],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    meta["portal_pid"]=process.pid
    for _ in range(30):
        status=portal_status(meta["port"])
        if status and status.get("session_id")==meta["session_id"] and status.get("online"):
            save(ACTIVE,meta)
            public=RUNTIME/"public-demo/active.json"
            if public.exists():
                shared=read(public);shared["expires_at"]=meta["expires_at"];save(public,shared)
            print("Browser portal ready; GPU session:",meta["session_id"],"expires:",meta["expires_at"],flush=True)
            return
        if process.poll() is not None:break
        time.sleep(.25)
    # Keep the old GPU alive until the replacement portal passes its health check.
    if process.poll() is None:process.terminate();process.wait(timeout=5)
    with (Path(previous["session"])/"portal.log").open("a") as log:
        rollback=subprocess.Popen([sys.executable,str(ROOT/"interactive.py"),"serve","--session",str(previous["session"]),"--port",str(previous["port"])],
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    previous["portal_pid"]=rollback.pid;save(ACTIVE,previous)
    raise RuntimeError("Replacement portal failed readiness; previous GPU portal restored")

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["start","serve","status","stop","refresh"])
    p.add_argument("--port",type=int,default=8768);p.add_argument("--session",type=Path)
    p.add_argument("--hours",type=int,default=4,choices=range(1,73));args=p.parse_args()
    if args.command=="start":start(args.port,args.hours)
    elif args.command=="serve":serve(args.session,args.port)
    elif args.command=="stop":stop()
    elif args.command=="refresh":refresh_portal(args.session)
    else:print(json.dumps(portal_status(args.port) or read(ACTIVE) or dict(online=False),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
