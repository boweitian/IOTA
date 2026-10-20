"""Replay raw scores and verify exhaustive sample assets, including baseline preservation."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
from sample_assets import ROOT,OLD,DEFAULT,geometry,exhaustive_bank,build_item,is_success
from experiment import read_records,grade,score_ifeval,parse_rubric
from engine import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--partial',action='store_true')
    args=p.parse_args();run=args.run;protocol=json.loads((ROOT/'inputs/protocol.json').read_text())
    rows={r['id']:r for r in protocol['records']};records={};manifests=[];reused=0;generated=0
    sources={}
    for out in sorted(run.glob('*/shard-*')):
        manifest=json.loads((out/'manifest.json').read_text());manifests.append(manifest)
        assert manifest['protocol']==protocol
        for name,digest in manifest['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        if (out/'judge_repair_protocol.json').exists():
            repair=json.loads((out/'judge_repair_protocol.json').read_text())
            assert repair['script_sha256']==hashlib.sha256((ROOT/'recover_writing_judge.py').read_bytes()).hexdigest()
        for key,r in read_records(out/'records.jsonl').items():
            assert key not in records;records[key]=r;row=rows[r['id']]
            if r.get('reused'):
                reused+=1;path=Path(r['source_file'])
                if path not in sources:sources[path]=path.read_text().splitlines()
                original=json.loads(sources[path][r['source_line']-1]);assert original['id']==r['id']
                assert original['action']==r['source_action'];assert original['reward']==r['reward']
                for field in ['text','token_ids','choice_log_probabilities','choice_probabilities']:
                    if field in original:assert r[field]==original[field]
            else:generated+=1
            if row['suite'] in ['gsm8k','bbh']:
                scored=grade(r['text'],row);assert bool(r['reward'])==scored['correct'];assert r['extracted']==scored['extracted']
            elif row['suite']=='ifeval':
                scored=score_ifeval(row,r['text']);assert r['reward']==scored['reward'];assert r['instruction_results']==scored['instruction_results']
            elif row['suite']=='truthfulqa':
                logp=np.asarray(r['choice_log_probabilities']);prob=np.exp(logp-logp.max());prob/=prob.sum()
                np.testing.assert_allclose(r['choice_probabilities'],prob,rtol=1e-12,atol=1e-14)
                assert abs(prob[np.asarray(row['targets']['labels'],dtype=bool)].sum()-r['reward'])<1e-12
            else:
                raw=(r.get('judge_retry') or r['judge_raw'])['text']
                scored=parse_rubric(raw,len(row['source_record']['checklist']))
                assert scored is not None;assert scored['reward']==r['reward'];assert scored['scores']==r['scores']
                if r.get('judge_repair'):
                    assert r['judge_repair']['script_sha256']==hashlib.sha256((ROOT/'recover_writing_judge.py').read_bytes()).hexdigest()
    required=[a['id'] for a in exhaustive_bank(geometry())]
    built=[build_item(r,records,required) for r in protocol['records']]
    if not args.partial:assert all(r['status']=='complete' for r in built)
    dictionary=[json.loads(line) for line in (run/'asset_bundle/sample_direction_dictionary.jsonl').read_text().splitlines()]
    assert dictionary==built
    for item in built:
        assert item['best_reward']>=item['baseline_reward']
        assert item['best_reward']==max(r['reward'] for r in item['evidence'].values())
        assert set(item['successful_directions'])=={name for name,r in item['evidence'].items()
            if is_success(rows[item['id']],r,item['evidence']['none'])}
    # Check saved baseline evidence is exactly the original no-intervention output, never a fitted steer.
    original={}
    for suite in protocol['suites']:
        original.update(read_records(OLD/suite/'records.jsonl'))
        for key,r in read_records(OLD/'validation_guard'/suite/'records.jsonl').items():
            original.setdefault(key,r)
    for row in protocol['records']:
        baseline=records[row['id'],'none'];assert baseline['reward']==original[row['id'],'none']['reward']
        if 'text' in baseline:assert baseline['text']==original[row['id'],'none']['text']
    matrix=np.load(run/'asset_bundle/response_matrix.npz')
    assert matrix['sample_ids'].tolist()==[r['id'] for r in protocol['records']]
    assert matrix['action_ids'].tolist()==required
    for i,row in enumerate(protocol['records']):
        for j,name in enumerate(required):
            found=(row['id'],name) in records;assert matrix['observed'][i,j]==found
            if found:assert matrix['reward'][i,j]==records[row['id'],name]['reward']
    with sqlite3.connect(run/'asset_bundle/sample_assets.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==len(rows)
        assert db.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]==len(records)
        assert db.execute('SELECT COUNT(*) FROM evaluations WHERE successful=1').fetchone()[0]==sum(len(r['successful_directions']) for r in built)
        assert db.execute('SELECT COUNT(*) FROM samples WHERE best_reward<baseline_reward').fetchone()[0]==0
    inverse=json.loads((run/'asset_bundle/direction_sample_index.json').read_text())['directions']
    for name,entry in inverse.items():
        assert set(entry['successful_samples'])=={r['id'] for r in built if name in r['successful_directions']}
        assert set(entry['improved_samples'])=={r['id'] for r in built if name in r['improving_directions']}
    cohorts=json.loads((run/'asset_bundle/execution_batches.json').read_text())['cohorts']
    assert sum(len(c['evidence_edges']) for c in cohorts.values())==sum(r['generated'] for r in records.values())
    for c in cohorts.values():assert len(c['sample_ids'])==c['batch_size']
    result=dict(passed=True,complete=sum(r['status']=='complete' for r in built),samples=len(built),
        required_settings=len(required),required_measurements=len(rows)*len(required),
        evidence_records=len(records),reused_records=reused,new_records=generated,
        exact_baselines_preserved=True,all_successes_preserved=True,score_replay_passed=True,
        monotonic_best_reward_verified=True,source_hashes_verified=True,
        sqlite_and_inverse_index_verified=True,original_generation_cohorts_preserved=True,
        missing_measurements=sum(len(r['missing_actions']) for r in built),
        geometry_audit=json.loads((run/'asset_bundle/geometry_construction_audit.json').read_text()))
    atomic_json(run/'evidence_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='geometry_audit'},indent=2))


if __name__=='__main__':main()
