"""Recover a WritingBench worker when the judge violates the fixed 1–10 scoring scale."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import sample_assets
from experiment import Execution,rubric_prompt,parse_rubric,append
from engine import atomic_json

original=Execution.judge_batch


def strict_retry(self,items):
    try:return original(self,items)
    except RuntimeError as exc:
        if str(exc)!='Malformed judge after retry; no invented score':raise
    for row,record in items:
        if (row['id'],record['action']) in self.records:continue
        generations=[];parsed=None
        for attempt in range(2):
            instruction=('\nSCORING SCALE: every score MUST be a number from 1 through 10 inclusive. '
                'The minimum score is 1, even when a criterion is completely failed. Never return 0. '
                'Return exactly one score per criterion in the original order, as a valid JSON scores array.')
            g=self.judge.generate([rubric_prompt(row,record['text'])+instruction],
                [np.zeros(self.judge.dim)],max_new_tokens=512+256*attempt)[0]
            generations.append(g);parsed=parse_rubric(g['text'],len(row['source_record']['checklist']))
            if parsed is not None:break
        if parsed is None:
            append(self.out/'judge_failures.jsonl',dict(id=row['id'],action=record['action'],generation=record,
                reason='Strict scale reminder still failed',repair_attempts=generations))
            raise RuntimeError('Strict judge repair failed; no invented score')
        record.update(parsed,judge_raw=generations[-1],judge_retry=None,judge_model=self.judge.meta,
            judge_repair=dict(reason='Judge returned score outside fixed 1–10 range',
                generations=generations,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.save(record)


def main():
    args=sys.argv[1:];suite=args[args.index('--suite')+1];shard=args[args.index('--shard')+1]
    out=sample_assets.DEFAULT/suite/f'shard-{shard}'
    atomic_json(out/'judge_repair_protocol.json',dict(reason='Actual judge output contained 0 despite fixed 1–10 scale',
        behavior='Re-evaluate saved generation with explicit minimum score reminder; no numerical clamping or invented scores',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    Execution.judge_batch=strict_retry;sample_assets.main()


if __name__=='__main__':main()
