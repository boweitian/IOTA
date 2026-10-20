"""Freeze official GLUE SST-2 train/validation samples and pre-existing geometry."""
import csv
import io
import json
import random
import shutil
import zipfile
from pathlib import Path
from sentiment_prepare import LAB,ROOT,sha,write,jsonl

OUT=ROOT/'runs/sentiment9b-glue-sst2-100-v1'
SOURCE='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'

def canonical(text):return ' '.join(text.lower().split())

def official_rows(archive,split):
    filename='SST-2/'+('train.tsv' if split=='train' else 'dev.tsv')
    with zipfile.ZipFile(archive) as z:
        parsed=list(csv.DictReader(io.StringIO(z.read(filename).decode()),delimiter='\t'))
    return [dict(id=f'glue:sst2:{split}:{i}',text=r['sentence'],label=int(r['label']),source_split=split,source_index=i) for i,r in enumerate(parsed)]

def main():
    out=OUT;out.mkdir(exist_ok=True,parents=True)
    if (out/'dataset-manifest.json').exists():
        manifest=json.loads((out/'dataset-manifest.json').read_text())
        for name,expected in manifest['files'].items():
            if sha(out/name)!=expected:raise ValueError('Frozen input changed: '+name)
        print('Frozen SST-2 pilot reused');return
    archive=out/'source/SST-2.zip'
    if not archive.exists():
        import urllib.request
        archive.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(SOURCE,timeout=60) as response:archive.write_bytes(response.read())
    prior=ROOT/'runs/sentiment9b-sst5-direction-sweep-v1/feasible-space.jsonl'
    excluded={canonical(json.loads(line)['text']) for line in prior.read_text().splitlines()}
    rng=random.Random(20261019);selected={};used=set(excluded);counts={}
    for split,per_class in [('train',35),('validation',15)]:
        rows=official_rows(archive,split);counts[split]=len(rows);chosen=[]
        for label in [0,1]:
            pool=[];seen=set()
            for row in rows:
                key=canonical(row['text'])
                if row['label']!=label or key in used or key in seen:continue
                seen.add(key);pool.append(row)
            chosen.extend(rng.sample(pool,per_class))
        rng.shuffle(chosen);selected[split]=chosen;used.update(canonical(r['text']) for r in chosen)
    assert counts==dict(train=67349,validation=872)
    assert len({canonical(r['text']) for rows in selected.values() for r in rows})==100
    jsonl(out/'train.jsonl',selected['train'])
    jsonl(out/'test-inputs.jsonl',[{k:v for k,v in r.items() if k!='label'} for r in selected['validation']])
    jsonl(out/'test-labels.jsonl',[dict(id=r['id'],label=r['label']) for r in selected['validation']])
    for name in ['search-geometry.npz','search-atlas.json','search-protocol.json','geometry-binding.json']:
        shutil.copy2(ROOT/'runs/sentiment9b-sst5-100-v1'/name,out/name)
    jsonl(out/'excluded-prior-texts.jsonl',[dict(text=text) for text in sorted(excluded)])
    names=['train.jsonl','test-inputs.jsonl','test-labels.jsonl','search-geometry.npz','search-atlas.json','search-protocol.json','geometry-binding.json','excluded-prior-texts.jsonl']
    manifest=dict(benchmark='GLUE / SST-2',short_name='SST-2',labels=['negative','positive'],display_labels=['负面','正面'],label_tokens=['A','B'],
        source_url=SOURCE,source_sha256=sha(archive),source_counts=counts,seed=20261019,
        train_count=70,test_count=30,train_per_class=35,test_per_class=15,
        train_source_split='official GLUE train',evaluation_source_split='official GLUE validation (dev.tsv)',
        train_ids=[r['id'] for r in selected['train']],test_ids=[r['id'] for r in selected['validation']],
        excluded_prior_samples=len(excluded),excluded_source_sha256=sha(prior),
        files={name:sha(out/name) for name in names},
        sampling='Fixed-seed class-balanced sample. Original text and labels retained. No duplicate sampled text, no exact text overlap with the inspected SST-5 pilot.',
        task_note='GLUE training data includes sentence fragments; original official rows are retained without rewriting or relabeling.',
        limitations='100-item pilot, not the official full GLUE test score. Thirty evaluation items are sampled from official validation because official test labels are hidden. Public-data pretraining exposure is not ruled out.')
    write(out/'dataset-manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
