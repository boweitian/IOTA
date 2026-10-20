"""Read local progress of the submitted GSM8K experiment without a scheduler query."""
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
run=ROOT/'runs/gsm-router-1000-v1'
protocol=json.loads((run/'protocol.json').read_text())
counts=[]
complete_train=0
for k in range(protocol['shards']):
    d=run/'train'/f'shard-{k}'
    p=d/'complete.json' if (d/'complete.json').exists() else d/'progress.json'
    counts.append(json.loads(p.read_text())['records'] if p.exists() else 0)
    complete_train+=int((d/'complete.json').exists())
complete_test=sum((run/'test'/f'shard-{k}'/'complete.json').exists() for k in range(protocol['shards']))
test_records=0
for path in (run/'test').glob('*/records.jsonl'):
    with path.open() as f:test_records+=sum(line.endswith('\n') for line in f)
fitted=(run/'training_summary.json').exists()
finished=(run/'complete.json').exists()
stage='complete' if finished else 'test' if fitted else 'fit' if complete_train==protocol['shards'] else 'measure_train'
jobs=json.loads((run/'jobs.json').read_text())
output=ROOT.parent/'output'
paths=list(output.glob(f"gsm1000-{jobs['train_array']}-*.log"))
paths += [output/f"gsm1000-fit-{jobs['fit']}.log",output/f"gsm1000-test-{jobs['test_and_report']}.log"]
errors=[]
for p in paths:
    if not p.exists():continue
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-5000));tail=f.read().decode(errors='replace')
    if any(x in tail for x in ['Traceback (most recent call last)','CUDA out of memory','CANCELLED AT']):
        errors.append(dict(file=p.name,tail=tail[-1800:]))
snapshot=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,
              train_cells=sum(counts),required_train_cells=76300,percent=round(100*sum(counts)/76300,2),
              train_shards_done=complete_train,test_shards_done=complete_test,shards=protocol['shards'],
              test_records=test_records,required_test_records=1200,
              fitted=fitted,finished=finished,errors=errors)
(run/'status.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
(run/'STATUS.md').write_text('# GSM8K 1,000 题实验进度\n\n'+
    f"更新时间：{snapshot['updated_utc']}。阶段：{stage}。\n\n"+
    f"训练方向遍历：{sum(counts):,} / 76,300（{snapshot['percent']}%）；已完成训练分片 {complete_train}/{protocol['shards']}。\n\n"+
    f"Router 训练完成：{fitted}；测试生成 {test_records}/1,200；测试分片完成 {complete_test}/{protocol['shards']}；最终报告完成：{finished}。\n\n"+
    '实验预先冻结 700 训练 / 300 测试；最终成绩见生成后的 RESULTS.md。\n')
print(json.dumps(snapshot,ensure_ascii=False))
