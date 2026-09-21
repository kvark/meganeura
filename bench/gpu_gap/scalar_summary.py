import collections
import json
from pathlib import Path
import statistics
import sys

for label in sys.argv[1:]:
    data = json.loads((Path(__file__).parent / label / 'result.json').read_text())
    records = data['records']
    print(label, data['device'], data['orientation'], len(records), 'cases',
          sum(r['failure'] is not None for r in records), 'failures')
    groups = collections.defaultdict(list)
    for record in records:
        if record['failure'] is None:
            groups[tuple(record['shape']), record['add']].append(record)
    for key, group in groups.items():
        rank = sorted(group, key=lambda r: statistics.median(r['gpu_us']))
        controls = [r for r in group if r['candidate'] in ('scalar-32', 'scalar-64')]
        old_layouts = [r for r in group if '-wg[16, 16]-' in r['candidate']
                       and r['candidate'].startswith(('scalar-[32, 32]', 'scalar-[64, 64]'))]
        best_control = min(controls, key=lambda r: statistics.median(r['gpu_us']))
        best_old = min(old_layouts, key=lambda r: statistics.median(r['gpu_us']))
        us = lambda r: round(statistics.median(r['gpu_us']), 3)
        print(key, 'control', us(best_control), 'old-search', us(best_old),
              'new', us(rank[0]), rank[0]['candidate'])
        for r in rank[:3]:
            print('  ', us(r), r['candidate'])
