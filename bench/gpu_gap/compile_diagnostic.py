"""Link a private Rust probe against the existing harness dependency build."""
import json
import os
from pathlib import Path
import subprocess
import sys

release = Path(os.environ.get('CARGO_TARGET_DIR', Path(__file__).parent / 'inferena/target')) / 'release'
fingerprints = release / '.fingerprint'

def dependency(meta, name):
    checksum = next(row[3] for row in meta['deps'] if row[1] == name)
    matches = [p for p in fingerprints.glob(f'*/lib-{name}')
               if int.from_bytes(bytes.fromhex(p.read_text().strip()), 'little') == checksum]
    assert len(matches) == 1, (name, matches, checksum)
    return matches[0]

runner = max(fingerprints.glob('inferena-meganeura-*/bin-inferena-meganeura.json'),
             key=lambda p: p.stat().st_mtime)
engine = dependency(json.loads(runner.read_text()), 'meganeura')
meta = json.loads(engine.with_suffix('.json').read_text())
engine_hash = engine.parent.name.rsplit('-', 1)[1]
command = ['rustc', '--edition=2024', '-O', sys.argv[1],
           '-L', f'dependency={release / "deps"}',
           '--extern', f'meganeura={release / "deps" / f"libmeganeura-{engine_hash}.rlib"}',
           '-o', sys.argv[2]]
for name in ['naga', 'blade_graphics', 'blade_macros', 'bytemuck', 'half', 'serde_json', 'egglog']:
    path = dependency(meta, name)
    crate_hash = path.parent.name.rsplit('-', 1)[1]
    suffix = 'so' if name == 'blade_macros' else 'rlib'
    command += ['--extern', f'{name}={release / "deps" / f"lib{name}-{crate_hash}.{suffix}"}']
blade = dependency(meta, 'blade_graphics')
ash = dependency(json.loads(blade.with_suffix('.json').read_text()), 'ash')
ash_hash = ash.parent.name.rsplit('-', 1)[1]
command += ['--extern', f'ash={release / "deps" / f"libash-{ash_hash}.rlib"}']
command += sys.argv[3:]
print(' '.join(command), flush=True)
subprocess.run(command, check=True)
