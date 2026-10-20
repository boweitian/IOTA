import json
import tempfile
import unittest
from pathlib import Path
from asset_index import generation_groups


class CohortTest(unittest.TestCase):
    def test_interrupted_grading_batch_uses_frozen_pending_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'manifest.json').write_text(json.dumps({'ids':['a','b','c','d']}))
            records=[dict(id=i,action='s',batch_seconds=t,batch_size=size,generated=True)
                for i,t,size in [('a',1.,4),('b',1.,4),('c',2.,2),('d',2.,2)]]
            path=out/'records.jsonl';path.write_text('\n'.join(map(json.dumps,records))+'\n')
            groups,inferred=generation_groups(path)
            self.assertEqual(groups['s',1.],['a','b','c','d'])
            self.assertEqual(groups['s',2.],['c','d'])
            self.assertEqual(inferred,{('s',1.)})

    def test_does_not_guess_when_record_order_disagrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'manifest.json').write_text(json.dumps({'ids':['a','b','c']}))
            path=out/'records.jsonl';path.write_text(json.dumps(dict(id='b',action='s',generated=True,batch_seconds=1.,batch_size=3))+'\n')
            with self.assertRaises(ValueError):generation_groups(path)


if __name__=='__main__':unittest.main()
