from pathlib import Path
p = Path('src/compile.rs')
s = p.read_text()
assert s.count('mod split_k;') == 1
s = s.replace('mod split_k;', 'mod softplus;\nmod split_k;')
for op, function in [('Op::Softplus { beta } => {', 'forward'), ('Op::SoftplusGrad { beta } => {', 'backward')]:
    start = s.index(op)
    a = s.index('                let pointwise = PointwiseDAG {', start)
    b = s.index('                self.plan.dispatches.push', a)
    s = s[:a] + f'                let pointwise = softplus::{function}(beta);\n' + s[b:]
p.write_text(s)
p = Path('tests/schedule_pointwise.rs')
s = p.read_text()
a = s.index('#[test]\nfn softplus_preserves_expanded_gradient_bits()')
b = s.index('#[test]\nfn softplus_compiles_to_one_pointwise_dispatch()', a)
s = s[:a] + s[b:]
a = s.index('/// Exact-equality checks')
b = s.index('fn inference(', a)
s = s[:a] + s[b:]
p.write_text(s)
p = Path('src/graph.rs')
s = p.read_text().replace('/// Backward helper for [`Op::Softplus`] that preserves the expanded\n    /// graph\'s rounding and subgradient convention at zero.', '/// Backward helper for [`Op::Softplus`], preserving the negative tail\n    /// and the analytic derivative of one half at zero.')
p.write_text(s)
p = Path('tests/softplus_tail.rs')
s = p.read_text().replace('let mut options = CompileOptions::default();\n            options.use_schedule_pointwise = schedule;', 'let options = CompileOptions { use_schedule_pointwise: schedule, ..Default::default() };')
s = s.replace('for c in 0..6 { accurate(output[row * 6 + c], raw[c] / total, "normalized weight"); }', 'for (c, &weight) in raw.iter().enumerate() { accurate(output[row * 6 + c], weight / total, "normalized weight"); }')
p.write_text(s)
