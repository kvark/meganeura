"""Bounded, warm-phase graphics capture using a frozen native runner."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).parent
label, model, binary, ept = sys.argv[1:]
destination = root / label
destination.mkdir()
env = dict(os.environ, MEGANEURA_DEVICE_ID='12036', MEGANEURA_FLASH_EPT_CAP=ept,
           MEGANEURA_TUNE='0', INFERENA_STRICT='1', INFERENA_WARMUP_RUNS='5',
           INFERENA_MEASUREMENT_RUNS='20', INFERENA_NSYS='markers',
           INFERENA_NGFX_PHASE='inference', FRAMEWORK_REV=Path(binary).name)
env.pop('INFERENA_PROFILE_DIR', None)
env.pop('INFERENA_INFERENCE_ONLY', None)
ngfx = os.environ.get('GAP_NGFX', shutil.which('ngfx'))
assert ngfx, 'Set GAP_NGFX to the Nsight Graphics executable'
command = [ngfx,
           '--activity', 'GPU Trace Profiler', '--exe', binary,
           '--dir', str(root / 'inferena'), '--args', model,
           '--output-dir', str(destination), '--start-with-ngfx-sdk',
           '--max-duration-ms', '200', '--limit-to-submits', '3',
           '--architecture', 'Blackwell GB20x', '--metric-set-id', '0',
           '--real-time-shader-profiler', '--auto-export',
           '--set-gpu-clocks', 'unaltered', '--collect-screenshot', '0',
           '--trace-timeout', '120']
(destination / 'manifest.json').write_text(json.dumps({'command':command, 'model':model,
    'binary':binary,'ept':ept,
    'flash_layout': {key: env.get(key) for key in ['MEGANEURA_FLASH_THREADS',
        'MEGANEURA_FLASH_KEYS', 'MEGANEURA_FLASH_INTERLEAVE']},
    'purpose':'GPU counters; not benchmark timing'}, indent=2))
with (destination / 'launcher.log').open('w') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
print(label, 'exit', result.returncode, flush=True)
sys.exit(result.returncode)
