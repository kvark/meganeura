import collections
import json
from pathlib import Path
import sys

for label in sys.argv[1:]:
    root = Path(__file__).parent / label
    print(label)
    for path in sorted(root.glob('*_meganeura.json')) + sorted(root.glob('*_pytorch.json')):
        data = json.loads(path.read_text())
        print(path.name, data.get('status'), data.get('timings'))
        for session in data.get('optimizer', {}).get('sessions', []):
            search = session.get('search')
            if not search:
                continue
            selected = search['selected']
            trials = search['trials']
            print(' ', session['mode'], 'trials', len(trials), 'selected', selected,
                  trials[selected].get('description'),
                  'qualification failures', sum(not t['outcome']['qualified'] for t in trials))
    for path in sorted((root / 'profiles').glob('*_inference_*.json')):
        data = json.loads(path.read_text())
        dispatches = data['profile']['dispatches']
        families = collections.defaultdict(float)
        for dispatch in dispatches:
            families[dispatch['family']] += dispatch['median_ms']
        print(' profile families, summed interval ms:', dict(families))
        for dispatch in sorted(dispatches, key=lambda x: x['median_ms'], reverse=True)[:6]:
            print('  ', round(dispatch['median_ms'], 4), dispatch['label'], dispatch['pipeline'])
