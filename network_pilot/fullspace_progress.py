"""Publish transparent progress without presenting partial accuracies as final."""
import argparse
import collections
import json
import time

from fullspace_detailed import OUT, atomic_write
from build_fullspace_view import DIST, REL


def update():
    counts=collections.Counter()
    for directory in (OUT/'outputs').glob('action-*'):
        counts[directory.name]=len(list(directory.glob('batch-*.jsonl')))
    completed=sum(counts.values())*16
    phase='generating';label='正在生成详细分析与最终答案'
    if (OUT/'summary.json').exists():phase='diagnosing';label='全部基准生成已完成，正在汇总可见错题的候选原因'
    if (OUT/'diagnosis-summary.json').exists():phase='auditing';label='正在独立核对全部输出、注入、评分与统计'
    final=DIST/'anli-fullspace.html'
    if (OUT/'audit.json').exists() and final.exists() and '<!--fullspace-complete-->' in final.read_text():
        phase='published';label='已完成并发布，独立核查通过'
    diagnosis_done=0
    if (OUT/'diagnosis').exists():
        for path in (OUT/'diagnosis').glob('rank-*.jsonl'):
            diagnosis_done+=len(path.read_text().splitlines())
    workers=[]
    for path in (OUT/'workers').glob('status-*.json'):
        try:workers.append(json.loads(path.read_text()))
        except (OSError,json.JSONDecodeError):pass
    value=dict(phase=phase,phase_label=label,completed_generations=completed,expected_generations=73200,
               completed_main_generations=sum(n for k,n in counts.items() if int(k.split('-')[1])<53)*16,
               completed_random_generations=sum(n for k,n in counts.items() if int(k.split('-')[1])>=53)*16,
               completed_configurations=sum(n==75 for n in counts.values()),configurations=61,
               diagnosis_records=diagnosis_done,active_worker_statuses=len(workers),updated_at=time.time())
    assets=DIST/REL;assets.mkdir(parents=True,exist_ok=True)
    atomic_write(assets/'progress.json',value);atomic_write(OUT/'progress.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    started=time.monotonic()
    while True:
        try:
            value=update();print(json.dumps(value),flush=True)
        except (OSError,json.JSONDecodeError) as error:
            print(json.dumps(dict(transient_read_error=str(error),retrying=True,updated_at=time.time())),flush=True)
            if not a.watch:raise
            time.sleep(5)
            continue
        if not a.watch or value['phase']=='published' or time.monotonic()-started>8*3600:break
        time.sleep(20)
