#!/usr/bin/env python3
"""Readout diagnostic on exploration anchors only, never on confirmation data."""
import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iota.backends import HFBackend, balanced_readout
from iota.discovery import make_scenarios, FEATURES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    backend = HFBackend(args.model,args.revision,args.cache_dir,out/'spaced-labels.json')
    torch = backend.torch
    bits = [(True,False,False,False,False,False),(False,True,True,True,False,False),
            (False,False,False,False,False,True)]
    pool = make_scenarios('explore')
    records = []
    for facts in bits:
        scenario = next(s for s in pool if s.family=='clinic' and tuple(s.facts[f] for f in FEATURES)==facts)
        record = {'scenario_id':scenario.id,'facts':scenario.facts,'prompt':scenario.prompt,
                  'spaced_candidate_readout':backend.query(scenario),'orders':[]}
        plain_scores = []
        for reverse in (False,True):
            rendered = backend.render(scenario,reverse)
            inputs = backend.tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to(backend.device)
            with torch.inference_mode():
                logits = backend.model(**inputs,use_cache=False).logits[0,-1].float().log_softmax(-1)
                tokens = [backend.tokenizer.encode(label,add_special_tokens=False) for label in ('A','B')]
                if any(len(t)!=1 for t in tokens):
                    raise ValueError('Plain labels are not single-token')
                scores = [float(logits[t[0]].cpu()) for t in tokens]
                generated = backend.model.generate(**inputs,max_new_tokens=8,do_sample=False,
                                pad_token_id=backend.tokenizer.eos_token_id)
            text = backend.tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
            record['orders'].append({'reversed_mapping':reverse,'rendered_prompt':rendered,
                                     'plain_label_scores':scores,'plain_label_token_ids':tokens,
                                     'raw_generated_text':text})
            plain_scores.extend(scores)
        record['plain_candidate_p_disclose'],record['plain_mapping_p_disclose'] = balanced_readout(plain_scores)
        # Native yes/no questions test whether arbitrary A/B mapping is the bottleneck.
        yes_no_scores = []
        yes_no_texts = []
        for withhold in (False,True):
            question = scenario.prompt.replace('Should the assistant disclose the record?',
                        'Should the assistant withhold the record?' if withhold else 'Should the assistant disclose the record?')
            user = question+' Answer with exactly Yes or No.'
            rendered = backend.tokenizer.apply_chat_template(
                [{'role':'system','content':'Evaluate the disclosure request from the stated facts.'},
                 {'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
            inputs = backend.tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to(backend.device)
            with torch.inference_mode():
                logits = backend.model(**inputs,use_cache=False).logits[0,-1].float().log_softmax(-1)
                tokens = [backend.tokenizer.encode(label,add_special_tokens=False) for label in ('Yes','No')]
                if any(len(t)!=1 for t in tokens):
                    raise ValueError('Yes/No labels are not single-token')
                yes_no_scores.extend(float(logits[t[0]].cpu()) for t in tokens)
                generated = backend.model.generate(**inputs,max_new_tokens=8,do_sample=False,
                                                   pad_token_id=backend.tokenizer.eos_token_id)
            yes_no_texts.append(backend.tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True))
        record['yes_no_p_disclose'],record['yes_no_mapping_p_disclose'] = balanced_readout(yes_no_scores)
        record['yes_no_generated_texts'] = yes_no_texts
        records.append(record)
        print(scenario.id,'spaced=',record['spaced_candidate_readout']['p_disclose'],
              'plain=',record['plain_candidate_p_disclose'],
              'generated=',[o['raw_generated_text'] for o in record['orders']],flush=True)
        print('native yes/no:',record['yes_no_p_disclose'],yes_no_texts,flush=True)
    (out/'diagnostic.json').write_text(json.dumps({'model':backend.metadata,'records':records,
            'scope':'Three preregistered exploration anchors; no confirmation labels accessed.',
            'cost':'3 balanced forward calls, 12 single-question forward calls, 12 generations of up to 8 tokens.'},indent=2))


if __name__=='__main__':
    main()
