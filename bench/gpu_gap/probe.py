"""Keep a standalone diagnostic's artifacts outside all source checkouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).parent
label, binary, driver = sys.argv[1:4]
directory = root / label
directory.mkdir()
path = (root / binary).resolve()
env = dict(os.environ, VK_ICD_FILENAMES=f'/usr/share/vulkan/icd.d/{driver}_icd.json')
env.pop('VK_DRIVER_FILES', None)
env.pop('LD_PRELOAD', None)
env.pop('MEGANEURA_GPU_CAPTURE', None)
manifest = dict(binary=str(path), binary_sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                driver=driver, arguments=sys.argv[4:], purpose='targeted kernel diagnostic')
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
with (directory / 'result.json').open('w') as out, (directory / 'runner.log').open('w') as err:
    result = subprocess.run([str(path), *sys.argv[4:]], env=env, stdout=out, stderr=err)
print(label, 'exit', result.returncode, flush=True)
sys.exit(result.returncode)
