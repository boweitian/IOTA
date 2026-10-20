"""Pinned official IFEval scorer with an unambiguous module name.

Both steering_lab and network_pilot contain experiment.py. GPU initialization
puts steering_lab first on sys.path, so importing that basename is ambiguous.
"""
import hashlib
import os
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor_deps'))
sys.path.insert(0,str(ROOT/'vendor'))
os.environ['NLTK_DATA']=str(ROOT/'nltk_data')


def score_ifeval(row,text):
    from instruction_following_eval.evaluation_lib import InputExample,test_instruction_following_strict
    from langdetect import DetectorFactory
    DetectorFactory.seed=0
    random.seed(hashlib.sha256(row['id'].encode()).hexdigest())
    inp=InputExample(key=row['source_index'],instruction_id_list=row['instruction_id_list'],
                     prompt=row['question'],kwargs=row['kwargs'])
    scored=test_instruction_following_strict(inp,{row['question']:text})
    return dict(reward=float(scored.follow_all_instructions),
                instruction_results=[bool(x) for x in scored.follow_instruction_list])
