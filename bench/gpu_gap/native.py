"""Private kernel ablation; fixed binary, full model outputs, no cohort writes."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).parent
label, model, device, ept = sys.argv[1:5]
binary = (Path(sys.argv[5]) if len(sys.argv) > 5 else root / 'native-ab062b8').resolve()
destination = root / label
destination.mkdir()
env = dict(os.environ, MEGANEURA_DEVICE_ID=device, MEGANEURA_FLASH_EPT_CAP=ept,
           MEGANEURA_TUNE='0', MEGANEURA_GPU_TIMING='1',
           MEGANEURA_GPU_CAPTURE=os.environ.get('MEGANEURA_GPU_CAPTURE', '1'),
           INFERENA_STRICT='1', INFERENA_WARMUP_RUNS='5',
           INFERENA_MEASUREMENT_RUNS='20', INFERENA_PROFILE_SAMPLES='3',
           INFERENA_PROFILE_DIR=str(destination / 'profiles'),
           FRAMEWORK_REV=binary.name.removeprefix('native-'))
env.pop('INFERENA_INFERENCE_ONLY', None)
if '--accelerated' in sys.argv:
    env['INFERENA_STRICT'] = '0'
if '--tune' in sys.argv:
    env.update(MEGANEURA_TUNE='1', INFERENA_TUNE_SECONDS='60', INFERENA_COMPILE_SECONDS='120')
manifest = dict(binary=str(binary), model=model, device=device, ept=ept,
                tuning='--tune' in sys.argv, precision='accelerated-f32' if '--accelerated' in sys.argv else 'strict-f32', purpose='physical layout ablation',
                environment={k:v for k,v in env.items() if k.startswith(('MEGANEURA_', 'INFERENA_', 'GAP_'))})
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
with (destination / f'{model}_meganeura.json').open('w') as out, (destination / 'runner.log').open('w') as log:
    result = subprocess.run([str(binary), model], env=env, cwd=root / 'inferena', stdout=out, stderr=log)
print(label, 'exit', result.returncode, flush=True)
sys.exit(result.returncode)
