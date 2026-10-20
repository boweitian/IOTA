"""Publish completed scan suites without changing scientific scan workers."""
import os
import shutil
import subprocess
import sys
import time

import router_direction_scan as scan


def main():
    published={s:(scan.DEFAULT/s/'complete.json').exists() for s in scan.SUITES}
    env=dict(os.environ,MPLCONFIGDIR='/tmp/iota-mpl',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    while True:
        status=scan.status(scan.DEFAULT)
        shutil.copy2(scan.DEFAULT/'status.json',scan.core.LAB/'dist/assets/router-atlas/status.json')
        if status['failures']:raise RuntimeError(status['failures'])
        changed=any(status['suites'][s]['complete'] and not published[s] for s in scan.SUITES)
        if changed or status['complete']:
            subprocess.run([sys.executable,'build_router_atlas.py'],check=True,env=env)
            subprocess.run([sys.executable,'audit_router_atlas.py'],check=True,env=env)
            for s in scan.SUITES:published[s]=status['suites'][s]['complete']
        if status['complete']:break
        time.sleep(45)


if __name__=='__main__':main()
