"""Compare Quest benchmark tensors with fresh, identically initialized CPU runs."""
import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Inferena's strict full-tensor policy, fixed independently of measured errors.
ATOL, RTOL = 1e-6, 1e-4


def read_tensor(root, record):
    data = (root / record['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['sha256']
    dtype = {'F32': '<f4', 'U32': '<u4'}[record['dtype']]
    array = np.frombuffer(data, dtype=dtype).copy()
    assert array.size == record['elements'] == np.prod(record['shape'])
    assert np.isfinite(array).all()
    return torch.from_numpy(array.reshape(record['shape']))


def compare(actual, reference):
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    assert a.shape == r.shape and a.size
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    diff = np.abs(a - r)
    max_error, max_ref = float(diff.max()), float(np.abs(r).max())
    rms_error, rms_ref = float(np.sqrt(np.mean(diff**2))), float(np.sqrt(np.mean(r**2)))
    max_bound, rms_bound = ATOL + RTOL * max_ref, ATOL + RTOL * rms_ref
    return dict(elements=a.size, finite=finite, max_error=max_error,
                max_reference=max_ref, rms_reference=rms_ref,
                max_bound=max_bound, rms_error=rms_error, rms_bound=rms_bound,
                elementwise_mismatches=int(np.count_nonzero(diff > ATOL + RTOL * np.abs(r))),
                passed=finite and max_error <= max_bound and rms_error <= rms_bound)


def aggregate(checks):
    values = list(checks)
    count = sum(v['elements'] for v in values)
    maximum = max(v['max_error'] for v in values)
    max_ref = max(v['max_reference'] for v in values)
    rms = (sum(v['rms_error']**2 * v['elements'] for v in values) / count)**0.5
    rms_ref = (sum(v['rms_reference']**2 * v['elements'] for v in values) / count)**0.5
    finite = all(v['finite'] for v in values)
    return dict(elements=count, max_error=maximum, max_bound=ATOL + RTOL * max_ref,
                rms_error=rms, rms_bound=ATOL + RTOL * rms_ref, finite=finite,
                passed=finite and maximum <= ATOL + RTOL * max_ref and rms <= ATOL + RTOL * rms_ref)


def inputs_for(kind, tensors):
    x = dict(tensors)
    def pop(name, shape):
        return x.pop(name).reshape(shape)
    if kind == 'sd_unet':
        result = dict(noisy_latent=pop('noisy_latent', (1, 4, 32, 32)),
                      timestep_embedding=pop('timestep_embedding', (1, 64)),
                      text_context=pop('text_context', (77, 768)))
        if 'noise_target' in x:
            result['noise_target'] = pop('noise_target', (1, 4, 32, 32))
    elif kind == 'resnet':
        result = dict(images=pop('image', (-1, 3, 224, 224)))
        if 'labels' in x:
            result['labels'] = x.pop('labels').long().flatten()
    elif kind == 'whisper':
        result = dict(input_features=pop('mel', (1, 80, 3000)))
    elif kind == 'causal_lm':
        tokens = pop('token_ids', (1, -1)).long()
        result = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens))
        if 'labels' in x:
            result['labels'] = pop('labels', (1, -1)).long()
    elif kind == 'smolvla':
        result = dict(noisy_actions=pop('noisy_actions', (1, -1, 32)),
                      timestep=pop('timestep', (1, 1, 1440)))
        kv = [x.pop(key) for key in sorted(x) if key.startswith('vlm_kv_layer_')]
        assert len(kv) == 8 and all(torch.equal(kv[0], value) for value in kv)
        result['vlm_kv'] = kv[0].reshape(1, 16, 320)
        if 'target_actions' in x:
            assert torch.count_nonzero(x.pop('target_actions')) == 0
    else:
        raise AssertionError(kind)
    assert not x, f'unconsumed inputs: {list(x)}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--inferena', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64',
                        help='CPU arithmetic; float64 provides the higher-precision oracle')
    parser.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
    args = parser.parse_args()
    sys.path.insert(0, str(args.inferena / 'frameworks/pytorch'))
    bench = importlib.import_module('bench')
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    name = args.capture.name
    spec = bench.MODEL_REGISTRY[name]
    kind = spec['type']
    reference_dtype = getattr(torch, args.dtype)
    if kind == 'causal_lm':
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig.from_dict(json.loads(args.config.read_text()))
        config.dtype = torch.float32
        config._attn_implementation = args.attention
        model = LlamaForCausalLM(config).float()
    else:
        model = bench.load_model(name, spec, 'cpu').float()
        if kind == 'whisper':
            model.set_attn_implementation(args.attention)
    model.to(dtype=reference_dtype)
    # Match the benchmark modes. These workloads have no active dropout.
    model.eval() if kind == 'resnet' else model.train()
    assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    modules = dict(model.named_modules())
    parameters = {}
    for torch_name, parameter in model.named_parameters():
        canonical = bench._gradient_parameter_name(kind, torch_name)
        assert canonical not in parameters
        parent, _, leaf = torch_name.rpartition('.')
        transpose = leaf == 'weight' and isinstance(modules[parent], torch.nn.Linear)
        parameters[canonical] = (parameter, transpose)
    report = dict(model=name, reference='fresh PyTorch CPU', torch=torch.__version__,
                  reference_dtype=args.dtype, attention=args.attention,
                  host=platform.platform(), threads=args.threads,
                  bench_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
                  comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy=dict(atol=ATOL, rtol=RTOL, metrics=['maximum absolute error', 'RMS error'],
                              scope='each full output/loss/gradient tensor',
                              elementwise_allclose='diagnostic only'), phases={})
    tag = f'{args.dtype}-{args.attention}'
    reference_dir = args.capture / f'cpu-{tag}'
    reference_dir.mkdir(exist_ok=True)
    for phase in ('inference', 'latency', 'training'):
        start = time.monotonic()
        manifest = json.loads((args.capture / f'{phase}.json').read_text())
        assert manifest['phase'] == phase and manifest['precision'] == 'strict-f32'
        assert set(parameters) == set(manifest['parameters']), (
            'parameter inventory mismatch', set(parameters) ^ set(manifest['parameters']))
        with torch.no_grad():
            for canonical, (parameter, transpose) in parameters.items():
                source = read_tensor(args.capture, manifest['parameters'][canonical])
                shape = tuple(reversed(parameter.shape)) if transpose else parameter.shape
                assert tuple(source.shape) in (tuple(shape), (parameter.numel(),)), canonical
                source = source.reshape(shape)
                parameter.copy_(source.T if transpose else source)
                roundtrip = parameter.T if transpose else parameter
                assert torch.equal(roundtrip.flatten(), source.flatten()), canonical
        inputs = inputs_for(kind, {k: read_tensor(args.capture, v) for k, v in manifest['inputs'].items()})
        inputs = {k: v.to(reference_dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        model.zero_grad(set_to_none=True)
        checks, refs = {}, {}
        with torch.set_grad_enabled(phase == 'training'):
            output = bench._benchmark_forward(kind, model, inputs)
            if phase == 'training':
                loss = bench._benchmark_loss(kind, output, inputs)
                loss.backward()
                refs['loss'] = loss.detach().numpy()
                checks['loss'] = compare(manifest['loss'], refs['loss'])
                gradients = {key: (p, tr) for key, (p, tr) in parameters.items() if p.grad is not None}
                assert set(gradients) == set(manifest['gradients']), (
                    'gradient inventory mismatch', set(gradients) ^ set(manifest['gradients']))
                for canonical, (parameter, transpose) in gradients.items():
                    grad = parameter.grad.T if transpose else parameter.grad
                    reference = grad.detach().contiguous().numpy()
                    refs['gradient:' + canonical] = reference
                    actual = read_tensor(args.capture, manifest['gradients'][canonical])
                    checks['gradient:' + canonical] = compare(actual, reference)
            else:
                assert len(manifest['outputs']) == 1
                reference = bench._benchmark_logits(kind, output).detach().numpy()
                refs['output:0'] = reference
                checks['output:0'] = compare(read_tensor(args.capture, manifest['outputs'][0]), reference)
        np.savez(reference_dir / f'{phase}.npz', **refs)
        report['phases'][phase] = dict(checks=checks, seconds=time.monotonic() - start,
                                      manifest_sha256=hashlib.sha256((args.capture / f'{phase}.json').read_bytes()).hexdigest(),
                                      reference_sha256=hashlib.sha256((reference_dir / f'{phase}.npz').read_bytes()).hexdigest(),
                                      parameter_tensors=len(parameters),
                                      compared_elements=sum(v['elements'] for v in checks.values()),
                                      passed=all(v['passed'] for v in checks.values()))
        if phase == 'training':
            combined = aggregate(v for k, v in checks.items() if k.startswith('gradient:'))
            report['phases'][phase]['all_gradients'] = combined
            report['phases'][phase]['passed'] &= combined['passed']
        print(f'{name} {phase}: {report["phases"][phase]["passed"]}', flush=True)
        for key, check in checks.items():
            if not check['passed']:
                print(key, check, flush=True)
        # Save partial evidence even if a later phase fails.
        report['passed'] = len(report['phases']) == 3 and all(v['passed'] for v in report['phases'].values())
        (args.capture / f'comparison-{tag}.json').write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
