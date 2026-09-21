import collections
import json
from pathlib import Path
import statistics
import sys

for label in sys.argv[1:]:
    data = json.loads((Path(__file__).parent / label / 'result.json').read_text())
    records = data['records']
    print(label, data['device'], data.get('orientation'), len(records), 'cases',
          sum(r['failure'] is not None for r in records), 'failures')
    groups = collections.defaultdict(list)
    for record in records:
        if record['failure'] is None:
            groups[tuple(record['shape']), record['add']].append(record)
        else:
            print('FAIL', record['shape'], record['add'], record['candidate'], record['failure'])
    for key, group in groups.items():
        rank = sorted(group, key=lambda r: statistics.median(r['gpu_us']))
        controls = [r for r in group if r['candidate'] in ('scalar-32', 'scalar-64', 'production-coop', 'partitioned-coop')]
        print(key)
        for r in [*controls, *rank[:3]]:
            print('  ', round(statistics.median(r['gpu_us']), 3),
                  round(statistics.median(r['wall_us']), 3), r['candidate'])
