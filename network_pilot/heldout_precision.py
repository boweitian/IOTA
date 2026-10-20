"""Frozen, paired replication of baseline vs precision:-2 on unused ANLI R3 test."""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from answer_format import parse_answer
from classification_generation import DEFAULT, SOURCE, ClassificationEngine, traced_generate
from sentiment_navigator import digest, read
from sentiment_prepare import jsonl, sha, write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/anli-r3-heldout-precision-v1'
WORKERS = 8
SOURCES = ['heldout_precision.py', 'classification_generation.py', 'classification_sweep.py',
           'answer_format.py', 'sentiment_navigator.py', 'sentiment_prepare.py',
           '../steering_lab/engine.py', '../steering_lab/concepts.py']


def canonical(text):
    return ' '.join(text.casefold().split())


def frozen_write(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen artifact changed: {path}'
    else:
        write(path, value)


def prepare():
    OUT.mkdir(exist_ok=True)
    for folder in ['outputs', 'workers', 'source-code']:
        (OUT / folder).mkdir(exist_ok=True)
    if (OUT / 'spec.json').exists():
        return load()
    parent = json.loads((DEFAULT / 'generation-spec.json').read_text())
    used = {}
    used_files = {}
    for path in sorted((ROOT / 'runs').rglob('*.jsonl')):
        if OUT in path.parents or '/source/' in str(path):
            continue
        with path.open() as stream:
            line = stream.readline()
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get('id', '')).startswith('anli:r3:') or not {'context', 'hypothesis'} <= row.keys():
                continue
            rows = [row] + [json.loads(line) for line in stream if line.strip()]
        used_files[str(path)] = sha(path)
        for row in rows:
            if str(row.get('id', '')).startswith('anli:r3:') and 'context' in row:
                if row['id'] in used:
                    assert (row['context'], row['hypothesis']) == (used[row['id']]['context'], used[row['id']]['hypothesis'])
                used[row['id']] = {k: row[k] for k in ['id', 'context', 'hypothesis']}
    premises = {canonical(row['context']) for row in used.values()}
    pairs = {(canonical(row['context']), canonical(row['hypothesis'])) for row in used.values()}
    archive = SOURCE / 'source/anli_v1.0.zip'
    assert sha(archive) == json.loads((SOURCE / 'dataset-manifest.json').read_text())['source_sha256']
    with zipfile.ZipFile(archive) as source:
        raw = [json.loads(line) for line in source.read('anli_v1.0/R3/test.jsonl').decode().splitlines()]
    rows, labels, excluded = [], [], []
    for index, row in enumerate(raw):
        identity = 'anli:r3:test:' + row['uid']
        why = ('previously_evaluated_id' if identity in used else
               'shared_premise_with_previous_experiment' if canonical(row['context']) in premises else
               'shared_pair_with_previous_experiment' if (canonical(row['context']), canonical(row['hypothesis'])) in pairs else None)
        if why:
            excluded.append(dict(id=identity, source_index=index, reason=why))
            continue
        rows.append(dict(id=identity, uid=row['uid'], source_index=index, source_split='official R3 test',
                         context=row['context'], hypothesis=row['hypothesis']))
        labels.append(dict(id=identity, gold='enc'.index(row['label']), reason=row.get('reason', '')))
    assert len(rows) == 1155 and len(excluded) == 45 and len(used) == 130
    jsonl(OUT / 'inputs.jsonl', rows)
    jsonl(OUT / 'labels-for-scorer.jsonl', labels)
    jsonl(OUT / 'prior-inputs.jsonl', [used[k] for k in sorted(used)])
    write(OUT / 'exclusions.json', excluded)
    manifest = dict(benchmark='ANLI R3 unused official test', kind='classification_generation',
                    label_tokens=['A', 'B', 'C'], display_labels=['蕴含', '信息不足', '矛盾'],
                    prompt_template=parent['prompt_template'], prompt_fields=['context', 'hypothesis'])
    write(OUT / 'model-input-manifest.json', manifest)
    native = np.load(DEFAULT / 'search-geometry.npz')['directions'].astype(float)
    shifts = np.stack([np.zeros(native.shape[1]), -2 * native[2]])
    # Identical vector and dtype to the 77/100 anchor replay, without fitting any new direction.
    assert np.array_equal(shifts[1], np.load(DEFAULT / 'precision-anchor-v1/shifts.npz')['shifts'][0])
    np.savez_compressed(OUT / 'shifts.npz', shifts=shifts)
    source_hashes = {}
    for relative in SOURCES:
        path = ROOT / relative
        target = OUT / 'source-code' / path.name
        shutil.copy2(path, target)
        source_hashes[relative] = sha(path)
    spec = dict(
        registered_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        benchmark='ANLI R3', model=parent['model'], parent_spec_sha256=digest(parent),
        geometry_sha256=sha(DEFAULT / 'search-geometry.npz'), archive_sha256=sha(archive),
        prompt_template=parent['prompt_template'], prompt_fields=['context', 'hypothesis'],
        system_prompt=__import__('concepts').SYSTEM,
        conditions=[dict(id='baseline', name='无干预', norm=0.),
                    dict(id='precision:-2', name='规则精确 −2×', norm=float(np.linalg.norm(shifts[1])))],
        selection='Only baseline and the already selected precision:-2. No direction, dose, prompt, or sample selection based on new accuracy.',
        eligible_count=len(rows), official_test_count=len(raw), excluded_count=len(excluded),
        prior_unique_items=len(used), selection_order='All eligible items in original official test order; no correctness or label balancing filters.',
        batch_size=4, workers=WORKERS, max_new_tokens=256, secondary_max_new_tokens=512,
        assignment='Consecutive batches of 4, batch_index modulo 8; baseline then precision:-2 on the exact same batch.',
        scoring='Primary: same 256-token full greedy generation and strict terminal 答案：A/B/C line parser as prior 77/100 experiment. Invalid or truncated replies count as incorrect.',
        length_control='Prespecified secondary 512-token comparison for BOTH conditions. Replay entire original batch if any member is truncated; require all original token prefixes to match exactly. Reuse original EOS-complete outputs. Never select a budget by accuracy.',
        precision='FP32, TF32 disabled, non-thinking, greedy, original engine, original system prompt.',
        intervention='Original precision:-2 vector at block index 16, final prefill position and final position of every decode forward; no new directions and no parameter updates.',
        reporting='Paired accuracy difference, Wilson 95% marginal intervals, exact two-sided McNemar/binomial discordance test, prespecified paired bootstrap 95% CI seed 2026102003; class-specific scores, invalid/truncated counts, fixed/damaged cases.',
        limitations='Fresh relative to this local experiment only. Public ANLI may have appeared in pretraining. Original test subset after excluding 45 exposed/shared-premise items, not the full 1200-item official leaderboard score. One selected intervention on one model/benchmark.',
        prior_input_files_sha256=used_files, source_sha256=source_hashes,
        files={name: sha(OUT / name) for name in ['inputs.jsonl', 'labels-for-scorer.jsonl', 'prior-inputs.jsonl',
                                                'exclusions.json', 'model-input-manifest.json', 'shifts.npz']})
    write(OUT / 'spec.json', spec)
    print(json.dumps(dict(registered=True, n=len(rows), excluded=len(excluded), spec_sha256=digest(spec)), ensure_ascii=False), flush=True)
    return spec, manifest, shifts


def load():
    spec = json.loads((OUT / 'spec.json').read_text())
    for name, value in spec['files'].items():
        assert sha(OUT / name) == value, name
    for relative, value in spec['source_sha256'].items():
        assert sha(ROOT / relative) == value, relative
        assert sha(OUT / 'source-code' / Path(relative).name) == value
    assert sha(DEFAULT / 'search-geometry.npz') == spec['geometry_sha256']
    manifest = json.loads((OUT / 'model-input-manifest.json').read_text())
    return spec, manifest, np.load(OUT / 'shifts.npz')['shifts']


def atomic_jsonl(path, rows):
    partial = path.with_suffix('.tmp')
    jsonl(partial, rows)
    partial.replace(path)


def worker(shard):
    spec, manifest, shifts = load()
    assert 0 <= shard < spec['workers']
    # This process never opens the scorer labels or annotator reasons.
    rows = read(OUT / 'inputs.jsonl')
    engine = ClassificationEngine(spec['model'], manifest)
    engine.meta.update(intervention=spec['intervention'], scoring=spec['scoring'],
                       job_id=os.environ.get('SLURM_JOB_ID'), registered_spec_sha256=digest(spec),
                       loaded_at=time.time())
    write(OUT / 'workers' / f'model-{shard}.json', engine.meta)
    owned = [i for i in range(0, len(rows), 4) if (i // 4) % spec['workers'] == shard]
    records = {}
    for condition in range(2):
        path = OUT / 'outputs' / f'condition-{condition}-shard-{shard}.jsonl'
        records[condition] = read(path) if path.exists() else []
        assert len(records[condition]) % 4 == 0 or len(records[condition]) == sum(len(rows[i:i+4]) for i in owned)
        expected = [r['id'] for i in owned for r in rows[i:i+4]]
        assert [v['id'] for v in records[condition]] == expected[:len(records[condition])]
        assert all(v['spec_sha256'] == digest(spec) for v in records[condition])
    for lo in owned:
        batch = rows[lo:lo+4]
        for condition in range(2):
            if batch[0]['id'] in {v['id'] for v in records[condition]}:
                continue
            questions = [engine.question(row) for row in batch]
            started = time.time()
            generated = traced_generate(engine, questions, shifts[condition], spec['max_new_tokens'])
            replayed = any(v['truncated'] for v in generated)
            extended = traced_generate(engine, questions, shifts[condition], spec['secondary_max_new_tokens']) if replayed else generated
            for row, question, value, ext in zip(batch, questions, generated, extended):
                assert ext['token_ids'][:len(value['token_ids'])] == value['token_ids'], (row['id'], condition, 'Prefix mismatch')
                def result(v):
                    return dict(**v, **parse_answer(v['text'], manifest['label_tokens'], v['truncated']))
                records[condition].append(dict(id=row['id'], source_index=row['source_index'], condition=condition,
                    spec_sha256=digest(spec), prompt_sha256=hashlib.sha256(engine.prefix(question).encode()).hexdigest(),
                    batch_start=lo, shard=shard, started_at=started, completed_at=time.time(),
                    primary=result(value), secondary=result(ext), secondary_replayed=replayed))
            atomic_jsonl(OUT / 'outputs' / f'condition-{condition}-shard-{shard}.jsonl', records[condition])
        status = dict(shard=shard, completed_pairs=len(records[0]), assigned_pairs=sum(len(rows[i:i+4]) for i in owned),
                      updated_at=time.time(), last_batch_start=lo)
        write(OUT / 'workers' / f'status-{shard}.json', status)
        print(json.dumps(status), flush=True)
    write(OUT / 'workers' / f'done-{shard}.json', dict(complete=True, spec_sha256=digest(spec), completed_at=time.time()))


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def paired_metrics(rows, labels, base, steered, budget):
    gold = np.array([r['gold'] for r in labels])
    results = []
    correctness = []
    for outputs in [base, steered]:
        values = [v[budget] for v in outputs]
        pred = np.array([-1 if v['prediction'] is None else v['prediction'] for v in values])
        ok = pred == gold
        correctness.append(ok)
        results.append(dict(n=len(rows), correct=int(ok.sum()), accuracy=float(ok.mean()),
            accuracy_95ci=wilson(int(ok.sum()), len(rows)), invalid=sum(not v['format_valid'] for v in values),
            truncated=sum(v['truncated'] for v in values),
            confusion=[[int(((gold == i) & (pred == j)).sum()) for j in [-1, 0, 1, 2]] for i in range(3)],
            per_class=[dict(label='ABC'[i], n=int((gold == i).sum()), correct=int(ok[gold == i].sum()), accuracy=float(ok[gold == i].mean())) for i in range(3)]))
    old, new = correctness
    fixed, damaged = int((~old & new).sum()), int((old & ~new).sum())
    discordant = fixed + damaged
    # The two-sided exact sign/binomial test equals exact McNemar's test.
    p_value = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(fixed, damaged)+1)) / 2**discordant) if discordant else 1.
    difference = new.astype(int) - old.astype(int)
    rng = np.random.default_rng(2026102003)
    frequencies = np.bincount(difference + 1, minlength=3)
    boot = rng.multinomial(len(rows), frequencies / len(rows), size=20000)
    boot_difference = (boot[:, 2] - boot[:, 0]) / len(rows)
    return dict(baseline=results[0], precision_minus2=results[1], fixed=fixed, damaged=damaged,
                both_correct=int((old & new).sum()), both_wrong=int((~old & ~new).sum()),
                accuracy_difference=float(difference.mean()), difference_95ci=np.quantile(boot_difference, [.025, .975]).tolist(),
                exact_mcnemar_p=p_value,
                fixed_ids=[r['id'] for r, a, b in zip(rows, old, new) if not a and b],
                damaged_ids=[r['id'] for r, a, b in zip(rows, old, new) if a and not b])


def summarize():
    spec, manifest, shifts = load()
    rows = read(OUT / 'inputs.jsonl')
    labels = read(OUT / 'labels-for-scorer.jsonl')
    assert [r['id'] for r in rows] == [r['id'] for r in labels]
    records = []
    for condition in range(2):
        mapped = {}
        for shard in range(spec['workers']):
            assert json.loads((OUT / 'workers' / f'done-{shard}.json').read_text())['complete']
            for value in read(OUT / 'outputs' / f'condition-{condition}-shard-{shard}.jsonl'):
                assert value['id'] not in mapped and value['condition'] == condition and value['spec_sha256'] == digest(spec)
                mapped[value['id']] = value
        assert set(mapped) == {r['id'] for r in rows}
        records.append([mapped[r['id']] for r in rows])
    for a, b in zip(*records):
        assert a['prompt_sha256'] == b['prompt_sha256'] and a['batch_start'] == b['batch_start'] and a['shard'] == b['shard']
    summaries = {budget: paired_metrics(rows, labels, *records, budget) for budget in ['primary', 'secondary']}
    per_item = []
    for i, (row, label) in enumerate(zip(rows, labels)):
        per_item.append(dict(**row, **{k:label[k] for k in ['gold', 'reason']},
                             outputs=[records[j][i] for j in range(2)]))
    jsonl(OUT / 'paired-results.jsonl', per_item)
    output_hashes = {str(path.relative_to(OUT)):sha(path) for path in sorted((OUT / 'outputs').glob('*.jsonl'))}
    summary = dict(complete=True, spec_sha256=digest(spec), n=len(rows), results=summaries,
                   actual_primary_generations=len(rows)*2,
                   actual_secondary_generations=sum(v['secondary_replayed'] for values in records for v in values),
                   output_sha256=output_hashes, paired_results_sha256=sha(OUT / 'paired-results.jsonl'),
                   completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), limitation=spec['limitations'])
    write(OUT / 'summary.json', summary)
    print(json.dumps({k:{n:v for n,v in value.items() if not n.endswith('_ids')} for k,value in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        worker(args.worker)
    elif args.summarize:
        summarize()
    else:
        parser.error('Choose --prepare, --worker, or --summarize')
