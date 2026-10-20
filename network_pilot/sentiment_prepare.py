"""Freeze a balanced 70/30 SST-5 pilot and the already discovered Search geometry."""
import argparse
import hashlib
import json
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent/'steering_lab'
SOURCE='https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
DEFAULT=ROOT/'runs/sentiment9b-sst5-100-v1'
LABELS=['very negative','negative','neutral','positive','very positive']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(path,rows):Path(path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))

def read_trees(text,split):
    rows=[];seen=set()
    for index,line in enumerate(text.splitlines()):
        if not line.strip():continue
        label=int(re.match(r'^\(([0-4])\s',line).group(1))
        tokens=re.findall(r'\([0-4] ([^()]*)\)',line)
        sentence=' '.join(tokens)
        for old,new in [('-LRB-','('),('-RRB-',')'),('\\/','/'),('\\*','*')]:sentence=sentence.replace(old,new)
        canonical=' '.join(sentence.lower().split())
        if not sentence or canonical in seen:continue
        seen.add(canonical)
        rows.append(dict(id=f'sst5:{split}:{index}',text=sentence,label=label,canonical=canonical,source_split=split))
    return rows

def balanced(rows,n,rng,excluded=()):
    selected=[];excluded=set(excluded)
    for label in range(5):
        pool=[r for r in rows if r['label']==label and r['canonical'] not in excluded]
        selected.extend(rng.sample(pool,n))
    rng.shuffle(selected)
    return [{k:v for k,v in row.items() if k!='canonical'} for row in selected]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'dataset-manifest.json').exists():
        manifest=json.loads((out/'dataset-manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if sha(out/name)!=digest:raise ValueError('Frozen dataset changed: '+name)
        print('Frozen dataset reused:',out);return
    source=out/'source';source.mkdir(exist_ok=True);archive=source/'trainDevTestTrees_PTB.zip'
    if not archive.exists():
        with urllib.request.urlopen(SOURCE,timeout=90) as response:archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        train_name=next(n for n in z.namelist() if n.endswith('/train.txt'))
        test_name=next(n for n in z.namelist() if n.endswith('/test.txt'))
        train=read_trees(z.read(train_name).decode(),'official_train')
        test=read_trees(z.read(test_name).decode(),'official_test')
    rng=random.Random(20261019)
    training=balanced(train,14,rng)
    testing=balanced(test,6,rng,{r['canonical'] for r in train})
    assert len(training)==70 and len(testing)==30
    assert not {r['text'].lower() for r in training}&{r['text'].lower() for r in testing}
    jsonl(out/'train.jsonl',training)
    jsonl(out/'test-inputs.jsonl',[{k:v for k,v in r.items() if k!='label'} for r in testing])
    jsonl(out/'test-labels.jsonl',[dict(id=r['id'],label=r['label']) for r in testing])
    geometry=LAB/'runs/qwen35-9b-studio-v2'
    for name in ['geometry.npz','atlas.json','protocol.json']:shutil.copy2(geometry/name,out/('search-'+name))
    atlas=json.loads((out/'search-atlas.json').read_text());protocol=json.loads((out/'search-protocol.json').read_text())
    if atlas['model']['model']!='Qwen/Qwen3.5-9B' or atlas['model']['layer_index']!=16:raise ValueError('Unexpected geometry binding')
    binding=dict(model=atlas['model'],protocol_sha256=protocol['sha256'],geometry_sha256=sha(out/'search-geometry.npz'),
        atlas_sha256=sha(out/'search-atlas.json'),source=str(geometry),construction='Search instruction-intent directions, frozen before SST sampling',
        geometry_depends_on=['model weights and revision','selected residual layer','positive/negative instructions',
            'construction contexts and activation readout','normalization and scale'],benchmark_labels_used_for_geometry=False)
    write(out/'geometry-binding.json',binding)
    files=['train.jsonl','test-inputs.jsonl','test-labels.jsonl','search-geometry.npz','search-atlas.json','search-protocol.json','geometry-binding.json']
    manifest=dict(benchmark='Stanford Sentiment Treebank SST-5',source_url=SOURCE,source_sha256=sha(archive),seed=20261019,
        train_count=70,test_count=30,labels=LABELS,train_per_class=14,test_per_class=6,
        sampling='whole sentence roots only; official train/test split; duplicate sentences excluded',
        train_ids=[r['id'] for r in training],test_ids=[r['id'] for r in testing],files={n:sha(out/n) for n in files},
        test_rule='Freeze policies on 70 training items before reading test inputs; test labels read only by final scorer.',
        limitations='Small stratified pilot, not a standard full-dataset score; model pretraining contamination is not ruled out.')
    write(out/'dataset-manifest.json',manifest)
    print(json.dumps(dict(out=str(out),train=70,test=30,labels=LABELS,model=atlas['model']['model'],geometry_sha256=binding['geometry_sha256']),ensure_ascii=False))

if __name__=='__main__':main()
