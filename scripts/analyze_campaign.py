#!/usr/bin/env python3
"""Retrospective frozen-prefix learning curves; no new inference or rule tuning."""
import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iota.backends import ReplayBackend
from iota.discovery import EvidenceMap,DiscoveryWalker,prediction_metrics,validate_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign')
    parser.add_argument('--budgets',type=int,nargs='+',default=[3,6,12,24])
    args = parser.parse_args()
    directory = Path(args.campaign)
    curves = []
    for path in sorted(directory.glob('*_map.json')):
        stem = path.name[:-9]
        full = EvidenceMap.load(path)
        confirmation = json.loads((directory/(stem+'_confirmation.json')).read_text())
        if confirmation['map_hash'] != full.content_hash:
            raise ValueError('Confirmation belongs to a different map')
        for obs in confirmation['observations']:
            validate_evidence(obs)
        for budget in args.budgets:
            if budget>sum(o['query_cost'] for o in full.observations):
                continue
            partial = DiscoveryWalker(ReplayBackend(full),full.strategy,full.seed,full.noise).run(budget)
            if [o['evidence_id'] for o in partial.observations] != [o['evidence_id'] for o in full.observations[:partial.version]]:
                raise ValueError('Prefix replay selected different evidence; refusing a leaky learning curve')
            predictions = [partial.predict(o['facts']) for o in confirmation['observations']]
            curves.append({'strategy':full.strategy,'seed':full.seed,'budget':budget,
                           'candidate':partial.best_rule()[0].description,
                           **prediction_metrics(predictions,confirmation['observations'])})
    output = {'analysis':'retrospective_frozen_prefixes','curves':curves,
              'new_inference_calls':0,
              'limitations':'Uses existing held-out observations after all exploration; does not tune rules or certify active superiority.'}
    (directory/'learning_curves.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(f'{len(curves)} frozen prefix evaluations; zero inference calls')


if __name__=='__main__':
    main()
