"""Prepare an isolated Inferena runner with full-tensor export for Android."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, f'benchmark patch point changed: {old!r}'
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inferena', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True,
                        help='Cargo.lock from a previously prepared Android runner')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    assert destination.is_relative_to(root / 'target'), 'write snapshots under target/'
    source = args.inferena.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'src').mkdir(exist_ok=True)
    originals = {}
    for path in (source / 'frameworks/meganeura/src').glob('*.rs'):
        originals[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, destination / 'src' / path.name)
    main_path = destination / 'src/main.rs'
    text = main_path.read_text()
    text = replace_once(text, 'mod initialization;', 'mod initialization;\nmod export;')
    text = replace_once(text, '    let mut config = session_config();',
                        '    export::set_graph(graph);\n    let mut config = session_config();')
    text = replace_once(text, '    let (warmups, samples) = benchmark_counts();',
                        '    set_inputs(session);\n    let mut exported = export::before(session, phase);\n'
                        '    let (warmups, samples) = benchmark_counts();')
    text = replace_once(text, '    BenchStats::from_samples(samples_ms, warmup)',
                        '    export::after(session, phase, &mut exported);\n'
                        '    BenchStats::from_samples(samples_ms, warmup)')
    main_path.write_text(text)
    cache = destination / 'src/search_cache.rs'
    cache.write_text(replace_once(cache.read_text(), '../../../Cargo.lock', '../Cargo.lock'))
    shutil.copyfile(Path(__file__).with_name('export.rs'), destination / 'src/export.rs')
    manifest = f'''[package]
name = "quest-full-qualification"
version = "0.1.0"
edition = "2024"
[workspace]
[[bin]]
name = "inferena-meganeura"
path = "src/main.rs"
[dependencies]
blade-macros = "0.3"
meganeura = {{ path = {json.dumps(root.as_posix())}, features = ["models"] }}
blade-graphics = {{ version = "0.9", git = "https://github.com/kvark/blade", rev = "56f05651cca73659130e924e497dda633f839d40" }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
sha2 = "0.10"
hex = "0.4"
env_logger = "0.11"
nvtx = "1.3"
'''
    (destination / 'Cargo.toml').write_text(manifest)
    lock = args.lock.read_text()
    if 'name = "quest-full-qualification"' not in lock:
        lock = replace_once(lock, 'name = "inferena-quest3s"', 'name = "quest-full-qualification"')
    (destination / 'Cargo.lock').write_text(lock)
    metadata = dict(inferena_revision=subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        original_sources=originals,
        prepared_sources={str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((destination / 'src').glob('*.rs')),
                                    destination / 'Cargo.toml', destination / 'Cargo.lock']},
        instrumentation='read complete weights/inputs before warmup; outputs/loss/gradients after measurement')
    (destination / 'preparation.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
