"""Loopback-only public gateway and an owned temporary HTTPS tunnel."""
import argparse
import base64
import hmac
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime/public-demo'
TUNNEL=ROOT.parents[2]/'tools/cloudflared/cloudflared'
APIS={'/api/studio/chat','/api/studio/compare','/api/studio/walk','/api/studio/direction','/api/route'}


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.chmod(0o600);temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def gateway(port,upstream,access,dist,require_auth=True):
    expected=None
    if require_auth:
        credentials=read(access)
        expected='Basic '+base64.b64encode((credentials['username']+':'+credentials['password']).encode()).decode()
    inference=threading.BoundedSemaphore(1);dist=Path(dist).resolve()
    jobs={};jobs_lock=threading.Lock()
    def prune_jobs():
        with jobs_lock:
            for key,value in list(jobs.items()):
                if value['expires_at']<time.time():jobs.pop(key,None)
    def execute_job(key,path,body):
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{upstream}'+path,data=body,headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=300) as response:
                    result=(response.status,response.read(),response.headers.get('Content-Type','application/json'))
            except urllib.error.HTTPError as error:
                result=(error.code,error.read(),error.headers.get('Content-Type','application/json'))
            except (OSError,TimeoutError):
                result=(503,json.dumps(dict(error='模型服务暂时无法连接，请稍后重试。'),ensure_ascii=False).encode(),'application/json; charset=utf-8')
            with jobs_lock:
                if key in jobs:jobs[key].update(result=result,expires_at=time.time()+900)
        finally:inference.release()
    class Handler(BaseHTTPRequestHandler):
        server_version='IOTA';sys_version=''
        def setup(self):super().setup();self.connection.settimeout(15)
        def log_message(self,*args):pass
        def respond(self,status,body,content_type='application/json; charset=utf-8',extra=None):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','private, no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','SAMEORIGIN')
            self.send_header('Referrer-Policy','same-origin')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers()
            if self.command!='HEAD':self.wfile.write(body)
        def authorized(self):
            if expected is None:return True
            supplied=self.headers.get('Authorization','')
            # Compare bytes so malformed non-ASCII headers cannot raise TypeError.
            if hmac.compare_digest(supplied.encode(),expected.encode()):return True
            self.respond(401,dict(error='请输入 IOTA 公网演示的用户名与访问密码。'),extra={'WWW-Authenticate':'Basic realm="IOTA demo", charset="UTF-8"'})
            return False
        def forward(self,path,body=None):
            headers={'Content-Type':'application/json'} if body is not None else {}
            request=urllib.request.Request(f'http://127.0.0.1:{upstream}'+path,data=body,headers=headers)
            try:
                with urllib.request.urlopen(request,timeout=300) as response:
                    self.respond(response.status,response.read(),response.headers.get('Content-Type','application/octet-stream'))
            except urllib.error.HTTPError as error:self.respond(error.code,error.read(),error.headers.get('Content-Type','application/json'))
            except (OSError,TimeoutError):self.respond(503,dict(error='模型服务暂时无法连接，请稍后重试。'))
        def do_GET(self):
            if not self.authorized():return
            parsed=urllib.parse.urlsplit(self.path)
            if parsed.scheme or parsed.netloc:self.respond(400,dict(error='Invalid path'));return
            path=urllib.parse.unquote(parsed.path)
            if path.startswith('/api/jobs/'):
                prune_jobs()
                key=path[len('/api/jobs/'):]
                with jobs_lock:job=jobs.get(key)
                if job is None:self.respond(410,dict(error='本次生成已过期，请重新提交。'))
                elif job.get('result') is not None:self.respond(*job['result'])
                else:self.respond(202,dict(pending=True))
                return
            if path=='/api/latest':
                # Private portal-wide trial results never cross into shared sessions.
                self.respond(200,dict(latest=None));return
            if path=='/api/status':self.forward('/api/status');return
            if path.startswith('/api/'):
                self.respond(404,dict(error='Unknown API'));return
            target=(dist/path.lstrip('/')).resolve()
            if path=='/':target=dist/'index.html'
            if not target.is_relative_to(dist) or not target.is_file():
                self.respond(404,dict(error='Not found'));return
            self.forward(urllib.parse.quote(path,safe='/')+('?' + parsed.query if parsed.query else ''))
        def do_HEAD(self):self.do_GET()
        def do_POST(self):
            if not self.authorized():return
            asynchronous=self.path.startswith('/api/async/')
            path='/api/'+self.path[len('/api/async/'):] if asynchronous else self.path
            if path not in APIS:self.respond(404,dict(error='Unknown API'));return
            origin=self.headers.get('Origin')
            allowed_origins=read(RUNTIME/'netlify-origins.json') if (RUNTIME/'netlify-origins.json').exists() else []
            if origin and origin not in allowed_origins and urllib.parse.urlsplit(origin).netloc.lower()!=self.headers.get('Host','').lower():
                self.respond(403,dict(error='Cross-origin submission is not allowed'));return
            if self.headers.get_content_type()!='application/json':
                self.respond(415,dict(error='Use application/json'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=120000:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('Expected an object')
                body.pop('display_in_browser',None)
            except (ValueError,TypeError,UnicodeError) as error:
                self.respond(400,dict(error=str(error)));return
            if not inference.acquire(blocking=False):
                self.respond(429,dict(error='模型正在处理另一条请求，请等待完成后再试。'),extra={'Retry-After':'10'});return
            encoded=json.dumps(body,ensure_ascii=False).encode()
            if asynchronous:
                prune_jobs();key=str(int(time.time()*1000))+'-'+secrets.token_hex(16)
                with jobs_lock:jobs[key]=dict(expires_at=time.time()+360,result=None)
                try:threading.Thread(target=execute_job,args=(key,path,encoded),daemon=True).start()
                except Exception:
                    with jobs_lock:jobs.pop(key,None)
                    inference.release();raise
                self.respond(202,dict(job_id=key,poll_url='/api/jobs/'+key));return
            try:self.forward(path,encoded)
            finally:inference.release()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print('IOTA gateway ready on loopback port',port,'password_protected=',require_auth,flush=True)
    server.serve_forever()


def stop():
    path=RUNTIME/'active.json'
    if not path.exists():print('No owned public demo');return
    meta=read(path)
    for key,signature in [('tunnel_pid',str(TUNNEL)),('gateway_pid',str(Path(__file__).resolve()))]:
        pid=meta.get(key)
        if not pid:continue
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            if signature.encode() in command:os.kill(pid,signal.SIGTERM)
        except (OSError,ProcessLookupError):pass
    path.unlink(missing_ok=True);print('Stopped owned public gateway and tunnel; local model session unchanged')


def start(port,upstream,require_auth=True):
    import re
    RUNTIME.mkdir(parents=True,exist_ok=True);RUNTIME.chmod(0o700)
    path=RUNTIME/'active.json'
    if path.exists():
        previous=read(path)
        if previous.get('url') and Path(f"/proc/{previous['tunnel_pid']}").exists():
            print(json.dumps(previous,ensure_ascii=False,indent=2));return
        raise RuntimeError('Previous incomplete public demo exists; inspect it before restarting')
    access=RUNTIME/'access.json'
    if require_auth and not access.exists():write(access,dict(username='iota',password=secrets.token_urlsafe(12)))
    with urllib.request.urlopen(f'http://127.0.0.1:{upstream}/api/status',timeout=15) as response:status=json.load(response)
    if not status.get('online'):raise RuntimeError('Connect the model before starting a public model demo')
    if not TUNNEL.is_file():raise RuntimeError('Verified cloudflared binary missing: '+str(TUNNEL))
    meta=dict(upstream=upstream,gateway_port=port,started_at=time.time(),expires_at=status.get('expires_at'),access_file=str(access) if require_auth else None,
              password_protected=require_auth,mode='temporary HTTPS; one public inference in flight; local latest results isolated')
    with (RUNTIME/'gateway.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'gateway','--port',str(port),'--upstream',str(upstream)]+([] if require_auth else ['--public-access']),
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    meta['gateway_pid']=process.pid;write(path,meta)
    headers={}
    if require_auth:headers['Authorization']='Basic '+base64.b64encode((read(access)['username']+':'+read(access)['password']).encode()).decode()
    ready=False
    for _ in range(40):
        if process.poll() is not None:break
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{port}/api/status',headers=headers)
            with urllib.request.urlopen(request,timeout=2) as response:ready=json.load(response).get('online',False)
            if ready:break
        except OSError:pass
        time.sleep(.25)
    if not ready:stop();raise RuntimeError('Public gateway failed its model readiness check')
    with (RUNTIME/'tunnel.log').open('wb') as log:
        tunnel=subprocess.Popen([str(TUNNEL),'tunnel','--no-autoupdate','--protocol','http2','--edge-ip-version','4','--url',f'http://127.0.0.1:{port}'],
                                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    meta['tunnel_pid']=tunnel.pid;write(path,meta)
    for _ in range(160):
        if tunnel.poll() is not None:break
        match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',(RUNTIME/'tunnel.log').read_text(errors='replace'))
        if match:
            meta['url']=match.group();write(path,meta);print(json.dumps(meta,ensure_ascii=False,indent=2));return
        time.sleep(.25)
    stop();raise RuntimeError('Tunnel did not return a public URL; inspect runtime/public-demo/tunnel.log')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','gateway','status','stop'])
    parser.add_argument('--port',type=int,default=8769);parser.add_argument('--upstream',type=int,default=8768)
    parser.add_argument('--public-access',action='store_true');args=parser.parse_args()
    if args.action=='gateway':gateway(args.port,args.upstream,RUNTIME/'access.json',ROOT/'dist',not args.public_access)
    elif args.action=='start':start(args.port,args.upstream,not args.public_access)
    elif args.action=='stop':stop()
    else:print(json.dumps(read(RUNTIME/'active.json') if (RUNTIME/'active.json').exists() else dict(online=False),ensure_ascii=False,indent=2))
