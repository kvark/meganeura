import collections
import json
from pathlib import Path
import sys

def seconds(d):
    return d['secs'] + d['nanos'] / 1e9

for label in sys.argv[1:]:
    path = next((Path(__file__).parent / label).glob('*_meganeura.json'))
    data = json.loads(path.read_text())
    print(label)
    for session in data.get('optimizer', {}).get('sessions', []):
        search = session.get('search')
        if not search:
            continue
        costs = collections.Counter()
        decisions = collections.Counter()
        for trial in search['trials']:
            for key in ['lowering_time', 'construction_time', 'initialization_time', 'state_copy_time', 'qualification_time']:
                costs[key] += seconds(trial[key])
            decisions[str(trial['outcome']['decision'])] += 1
        print(session['mode'], len(search['trials']), 'selected', search['selected'],
              'prepare', round(seconds(search['preparation_time']), 3),
              'total', round(seconds(search['elapsed']), 3),
              'components', {k: round(v, 3) for k,v in costs.items()},
              'decisions', dict(decisions))
