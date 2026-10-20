"""GPU worker for the browser portal. Shared-filesystem RPC needs no GPU SSH login."""
import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from engine import atomic_json
from serve import LiveRouter
from studio import CharacterStudio

def main():
    p=argparse.ArgumentParser();p.add_argument("--session",required=True,type=Path)
    p.add_argument("--run",required=True,type=Path);args=p.parse_args();session=args.session.resolve()
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status=dict(online=False,state="loading",updated_at=time.time())
    atomic_json(session/"worker.json",status)
    try:
        run=args.run.resolve()
        # Only the original 0.8B run has frozen benchmark-router policies; studio-only runs skip the router.
        if (run/"objective_policy.json").exists():
            live=LiveRouter(run,"cuda",256)
            studio=CharacterStudio(run,"cuda",256,engine=live.engine,lock=live.lock)
        else:
            live=None;studio=CharacterStudio(run,"cuda",256)
        status=dict(online=True,state="ready",studio=studio.metadata(),updated_at=time.time())
        atomic_json(session/"worker.json",status);print("GPU EDITOR READY",datetime.now(timezone.utc).isoformat(),flush=True)
        while not stopping:
            for path in sorted((session/"requests").glob("*.json")):
                if stopping:break
                active=path.with_suffix(".processing")
                path.rename(active)
                status.update(state="busy",updated_at=time.time());atomic_json(session/"worker.json",status)
                try:
                    request=json.loads(active.read_text())
                    body=request["body"]
                    if request["path"]=="/api/studio/compare": result=studio.compare(body.get("profile"),body.get("scene"))
                    elif request["path"]=="/api/studio/walk": result=studio.walk(body.get("profile"),body.get("scenes"))
                    elif request["path"]=="/api/studio/direction": result=studio.direction(body.get("axis"))
                    elif request["path"]=="/api/studio/chat": result=studio.chat(body.get("profile"),body.get("messages"),body.get("binding"))
                    elif request["path"]=="/api/route":
                        if live is None:raise ValueError("This model session has no benchmark router")
                        result=live.route(body.get("question"),body.get("domain"))
                    else:raise ValueError("Unknown interactive operation")
                    response=dict(status=200,body=result)
                except (ValueError,TypeError,KeyError) as error:
                    response=dict(status=400,body=dict(error=str(error)))
                except Exception as error:
                    response=dict(status=500,body=dict(error=str(error)))
                atomic_json(session/"responses"/(active.stem+".json"),response)
                active.unlink(missing_ok=True)
                status.update(state="ready",updated_at=time.time());atomic_json(session/"worker.json",status)
            status.update(studio=studio.metadata(),updated_at=time.time());atomic_json(session/"worker.json",status)
            time.sleep(.25)
    finally:
        status.update(online=False,state="stopped",updated_at=time.time());atomic_json(session/"worker.json",status)

if __name__=="__main__":main()
