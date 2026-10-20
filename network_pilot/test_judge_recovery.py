import json
import tempfile
import unittest
from pathlib import Path
from experiment import Execution
from recover_writing_judge import strict_retry


class FakeJudge:
    meta={'test':'fake'}
    dim=2
    def __init__(self,repairable):self.calls=0;self.repairable=repairable
    def generate(self,questions,shifts,**kwargs):
        self.calls+=1;score=1 if self.repairable and self.calls>=3 else 0
        return [{'text':json.dumps({'scores':[score],'reason':'criterion failed'})} for q in questions]


class JudgeRecoveryTest(unittest.TestCase):
    def items(self):
        row=dict(id='x',question='brief',source_record=dict(checklist=[dict(name='c',criteria_description='d')]))
        return [(row,dict(id='x',action='a',text='raw response'))]

    def test_reasks_actual_judge_instead_of_clamping_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Execution(None,Path(tmp),{});run.judge=FakeJudge(True)
            strict_retry(run,self.items());record=run.records['x','a']
            self.assertEqual(record['scores'],[1]);self.assertEqual(run.judge.calls,3)
            self.assertEqual(json.loads(record['judge_raw']['text'])['scores'],[1])
            failure=json.loads((Path(tmp)/'judge_failures.jsonl').read_text())
            self.assertEqual(json.loads(failure['judge']['text'])['scores'],[0])
            self.assertIn('judge_repair',record)

    def test_repeated_invalid_scores_remain_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Execution(None,Path(tmp),{});run.judge=FakeJudge(False)
            with self.assertRaisesRegex(RuntimeError,'Strict judge repair failed'):strict_retry(run,self.items())
            self.assertNotIn(('x','a'),run.records)


if __name__=='__main__':unittest.main()
