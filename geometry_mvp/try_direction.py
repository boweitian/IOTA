"""A small courage-knob usability check; raw generations, no personality scores."""
import argparse
from pathlib import Path
import numpy as np
from engine import SteeringEngine,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    e=SteeringEngine();ds=e.load_directions(Path(__file__).parent/'data/directions.npz')
    prompts=[
        '你有机会承担一个有难度的新任务，也可以选择不参加。你没有十足把握，但风险可控。你会选择尝试还是放弃？请先明确说出你的选择，再解释原因。',
        '有一个新机会，但你不确定自己能做好。请用第一人称表达你的态度，先说你愿不愿意尝试。',
        '大家都在等待有人提出一个不同意见。你担心被否定，但你有自己的理由。你会怎么做？请直接说明你的选择。',
    ]
    strengths=[-6,-4,-2,0,2,4,6]
    rows=[]
    for prompt in prompts:
        outputs=e.generate_batch(prompt,[s*ds['courage'] for s in strengths])
        for s,o in zip(strengths,outputs):
            rows.append(dict(prompt=prompt,strength=s,**o));print(s,o['text'],flush=True)
    atomic_json(args.output,dict(model=e.metadata,records=rows))
if __name__=='__main__':main()
