"""Independently check source isolation, prompts, token outputs and held-out scores."""
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from heldout_precision import OUT, ROOT, SOURCE, DEFAULT, canonical
from sentiment_navigator import digest, read
from sentiment_prepare import sha, write


def independently_parse(text, truncated):
    if truncated:
        return None
    lines = [line.strip() for line in text.strip().splitlines()]
    if len(lines) < 2 or not ''.join(lines[:-1]):
        return None
    if not re.fullmatch(r'答案[：:]\s*[ABC]', lines[-1]):
        return None
    if sum(bool(re.fullmatch(r'答案[：:]\s*[A-Z]', line)) for line in lines) != 1:
        return None
    return 'ABC'.index(lines[-1][-1])


def audit():
    from transformers import AutoTokenizer

    spec = json.loads((OUT / 'spec.json').read_text())
    registered_at = datetime.fromisoformat(spec['registered_at_utc'].replace('Z', '+00:00')).timestamp()
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['complete'] and summary['spec_sha256'] == digest(spec)
    for name, value in spec['files'].items():
        assert sha(OUT / name) == value, name
    for relative, value in spec['source_sha256'].items():
        assert sha(OUT / 'source-code' / Path(relative).name) == value
        assert sha(ROOT / relative) == value
    for name, value in spec['prior_input_files_sha256'].items():
        assert sha(name) == value
    assert sha(OUT / 'paired-results.jsonl') == summary['paired_results_sha256']
    archive = SOURCE / 'source/anli_v1.0.zip'
    assert sha(archive) == spec['archive_sha256']
    with zipfile.ZipFile(archive) as source:
        raw = [json.loads(line) for line in source.read('anli_v1.0/R3/test.jsonl').decode().splitlines()]
    previous = read(OUT / 'prior-inputs.jsonl')
    previous_ids = {r['id'] for r in previous}
    previous_premises = {canonical(r['context']) for r in previous}
    inputs = read(OUT / 'inputs.jsonl')
    labels = read(OUT / 'labels-for-scorer.jsonl')
    rows = read(OUT / 'paired-results.jsonl')
    eligible = [i for i, r in enumerate(raw) if 'anli:r3:test:'+r['uid'] not in previous_ids and canonical(r['context']) not in previous_premises]
    assert [r['source_index'] for r in inputs] == eligible
    assert len(inputs) == len(labels) == len(rows) == 1155
    assert {r['id'] for r in inputs}.isdisjoint(previous_ids)
    assert {canonical(r['context']) for r in inputs}.isdisjoint(previous_premises)
    assert all(set(r) == {'id', 'uid', 'source_index', 'source_split', 'context', 'hypothesis'} for r in inputs)
    for inp, label, row in zip(inputs, labels, rows):
        source = raw[inp['source_index']]
        assert inp['id'] == label['id'] == row['id'] == 'anli:r3:test:'+source['uid']
        assert (inp['context'], inp['hypothesis']) == (source['context'], source['hypothesis'])
        assert label['gold'] == row['gold'] == 'enc'.index(source['label'])
        assert label['reason'] == row['reason'] == source.get('reason', '')
    parent = json.loads((DEFAULT / 'generation-spec.json').read_text())
    assert digest(parent) == spec['parent_spec_sha256']
    assert parent['prompt_template'] == spec['prompt_template']
    assert spec['prompt_fields'] == ['context', 'hypothesis']
    original = np.load(DEFAULT / 'search-geometry.npz')['directions'].astype(float)
    shifts = np.load(OUT / 'shifts.npz')['shifts']
    assert np.array_equal(shifts[0], np.zeros(4096))
    assert np.array_equal(shifts[1], -2*original[2])
    tokenizer = AutoTokenizer.from_pretrained(spec['model']['model_path'], local_files_only=True)
    config = json.loads((Path(spec['model']['model_path']) / 'config.json').read_text())
    eos = config.get('eos_token_id', config.get('text_config', {}).get('eos_token_id'))
    eos = eos if isinstance(eos, list) else [eos]
    assert eos and None not in eos
    matrices = {budget: np.full((len(rows), 2), -1, dtype=int) for budget in ['primary', 'secondary']}
    source_outputs = {}
    for condition in range(2):
        for shard in range(8):
            name = f'outputs/condition-{condition}-shard-{shard}.jsonl'
            assert sha(OUT / name) == summary['output_sha256'][name]
            for record in read(OUT / name):
                assert (record['id'], condition) not in source_outputs
                source_outputs[record['id'], condition] = record
    model_devices = set()
    for shard in range(8):
        model = json.loads((OUT / 'workers' / f'model-{shard}.json').read_text())
        assert model['model'] == spec['model']['model'] and model['revision'] == spec['model']['model_revision']
        assert model['dtype'] == 'torch.float32' and model['layer_index'] == 16 and not model['thinking']
        assert model['system_prompt'] == spec['system_prompt']
        assert model['chat_template_sha256'] == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        assert model['registered_spec_sha256'] == digest(spec)
        assert model['loaded_at'] > registered_at
        model_devices.add(model['device'])
    actual_extensions = 0
    for i, row in enumerate(rows):
        assert len(row['outputs']) == 2
        question = spec['prompt_template'].format(context=json.dumps(row['context'], ensure_ascii=False),
                                                  hypothesis=json.dumps(row['hypothesis'], ensure_ascii=False))
        prefix = tokenizer.apply_chat_template([dict(role='system', content=spec['system_prompt']),
                    dict(role='user', content=question)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_hash = hashlib.sha256(prefix.encode()).hexdigest()
        for condition, record in enumerate(row['outputs']):
            assert record == source_outputs[row['id'], condition]
            assert record['spec_sha256'] == digest(spec) and record['prompt_sha256'] == prompt_hash
            assert record['condition'] == condition and record['batch_start'] == i//4*4 and record['shard'] == (i//4)%8
            assert registered_at < record['started_at'] <= record['completed_at']
            primary, secondary = record['primary'], record['secondary']
            if record['secondary_replayed']:
                actual_extensions += 1
                assert secondary['token_ids'][:len(primary['token_ids'])] == primary['token_ids']
            else:
                assert primary == secondary and not primary['truncated']
            for budget, maximum in [('primary', 256), ('secondary', 512)]:
                value = record[budget]
                assert value['new_tokens'] == len(value['token_ids']) <= maximum
                assert value['truncated'] == (len(value['token_ids']) == maximum and value['token_ids'][-1] not in eos)
                assert value['token_ids'][-1] in eos or value['truncated']
                assert tokenizer.decode(value['token_ids'], skip_special_tokens=True) == value['text']
                pred = independently_parse(value['text'], value['truncated'])
                assert value['prediction'] == pred and value['format_valid'] == (pred is not None)
                matrices[budget][i, condition] = -1 if pred is None else pred
                trace = value['steering_trace']
                norm = float(np.linalg.norm(shifts[condition]))
                assert abs(trace['requested_norm']-norm) < 1e-6
                assert max(abs(trace[key]-norm) for key in ['observed_min', 'observed_max']) < 1e-3
                assert trace['forward_calls'] > 0
    gold = np.array([r['gold'] for r in rows])
    for budget, matrix in matrices.items():
        metric = summary['results'][budget]
        ok = matrix == gold[:, None]
        fixed = int((~ok[:, 0] & ok[:, 1]).sum())
        damaged = int((ok[:, 0] & ~ok[:, 1]).sum())
        assert fixed == metric['fixed'] and damaged == metric['damaged']
        assert metric['both_correct'] == int(ok.all(1).sum()) and metric['both_wrong'] == int((~ok).all(1).sum())
        assert abs(metric['accuracy_difference'] - (fixed-damaged)/len(rows)) < 1e-12
        discordant = fixed+damaged
        p = min(1., math.fsum(math.exp(math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2)) for k in range(min(fixed,damaged)+1))*2) if discordant else 1.
        assert abs(p-metric['exact_mcnemar_p']) < 1e-10
        for condition, name in enumerate(['baseline', 'precision_minus2']):
            score = metric[name]
            assert score['correct'] == int(ok[:, condition].sum())
            assert abs(score['accuracy']-ok[:, condition].mean()) < 1e-12
            assert score['invalid'] == int((matrix[:, condition] < 0).sum())
            assert score['truncated'] == sum(r['outputs'][condition][budget]['truncated'] for r in rows)
            for j, category in enumerate(score['per_class']):
                assert category['n'] == int((gold == j).sum())
                assert category['correct'] == int(ok[gold == j, condition].sum())
        assert metric['fixed_ids'] == [r['id'] for r, a, b in zip(rows, ok[:, 0], ok[:, 1]) if not a and b]
        assert metric['damaged_ids'] == [r['id'] for r, a, b in zip(rows, ok[:, 0], ok[:, 1]) if a and not b]
    assert actual_extensions == summary['actual_secondary_generations']
    assert summary['actual_primary_generations'] == 2310
    result = dict(passed=True, n=len(rows), model_devices=sorted(model_devices), summary_sha256=sha(OUT/'summary.json'),
                  spec_sha256=digest(spec), checks=[
                      'All 1155 eligible official R3 test items, excluding every prior UID and normalized premise',
                      'Original texts and labels match archived official data; only context/hypothesis enter model prompts',
                      'Same complete chat prompt, batch membership and model for both conditions',
                      'Original precision:-2 vector; actual observed zero/nonzero injection norms checked on every generation',
                      'Full token outputs independently decoded, EOS/truncation checked, terminal labels independently parsed',
                      'All paired accuracy, class scores, invalid counts, fixed/damaged cases and exact p-values independently reconstructed',
                      '512-token replays exactly preserve 256-token prefixes; cached EOS records distinguished from new generations',
                      'Frozen input/source hashes and all original experimental input hashes unchanged'])
    write(OUT / 'audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    audit()
