"""Capture full benchmark tensors on Android and qualify them on PyTorch CPU."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODELS = ['StableDiffusion', 'ResNet-50', 'SmolVLA', 'SmolLM2-135M', 'Whisper-tiny']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--inferena', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    adb = ['adb', '-s', args.serial]
    remote = f'/data/local/tmp/meganeura/qualification-{int(time.time())}'
    executable = remote + '-runner'
    source = output / 'source'
    source.mkdir()
    # Freeze the exact CPU architecture and exporter alongside the evidence.
    snapshot = source / 'inferena/frameworks/pytorch'
    snapshot.mkdir(parents=True)
    for name in ('bench.py', 'execution.py', 'budget.py'):
        shutil.copyfile(args.inferena / 'frameworks/pytorch' / name, snapshot / name)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copyfile(path, source / path.name)
    shutil.copytree(args.runner / 'src', source / 'runner/src')
    for name in ('Cargo.toml', 'Cargo.lock', 'preparation.json'):
        shutil.copyfile(args.runner / name, source / 'runner' / name)
    patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD'])
    (source / 'meganeura.patch').write_bytes(patch)
    metadata = dict(
        meganeura_revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        serial=args.serial, remote_directory=remote, models=args.models,
        environment=dict(INFERENA_STRICT='1', MEGANEURA_TUNE='0',
                         INFERENA_WARMUP_RUNS='1', INFERENA_MEASUREMENT_RUNS='1'),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob('*') if p.is_file()},
    )
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2))
    subprocess.run(adb + ['push', str(args.binary), executable], check=True)
    subprocess.run(adb + ['shell', f'chmod +x {executable}'], check=True)
    config = output / 'smollm-config.json'
    if 'SmolLM2-135M' in args.models:
        subprocess.run(adb + ['pull', '/data/local/tmp/meganeura/models/SmolLM2-135M/config.json', str(config)], check=True)
    results = {}
    for model in args.models:
        print(f'Capturing {model}', flush=True)
        environment = ' '.join(f'{k}={v}' for k, v in metadata['environment'].items())
        command = f'QUALIFICATION_DIR={remote}/{model} {environment} {executable} {model}'
        with (output / f'{model}-gpu.log').open('wb') as log:
            subprocess.run(adb + ['shell', command], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(adb + ['pull', f'{remote}/{model}', str(output / model)], check=True)
        print(f'Qualifying {model} against CPU float64', flush=True)
        with (output / f'{model}-cpu.log').open('wb') as log:
            result = subprocess.run(
                [sys.executable, str(source / 'compare.py'), str(output / model),
                 '--inferena', str(source / 'inferena'), '--config', str(config), '--dtype', 'float64'],
                stdout=log, stderr=subprocess.STDOUT)
        report_path = output / model / 'comparison-float64-eager.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        results[model] = dict(passed=result.returncode == 0 and report.get('passed', False),
                              report=str(report_path.relative_to(output)))
        print(model, results[model], flush=True)
        summary = dict(models=results, complete=len(results) == len(args.models),
                       passed=len(results) == len(args.models) and all(v['passed'] for v in results.values()))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
