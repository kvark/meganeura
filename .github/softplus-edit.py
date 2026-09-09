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
s = p.read_text()
old = '/// Backward helper for [`Op::Softplus`] that preserves the expanded\n    /// stable identity\'s evaluation order.'
assert s.count(old) == 1
s = s.replace(old, '/// Analytic sigmoid derivative for [`Op::Softplus`], preserving the\n    /// negative tail and the derivative of one half at zero.')
p.write_text(s)
p = Path('tests/softplus_tail.rs')
s = p.read_text().replace('let mut options = CompileOptions::default();\n            options.use_schedule_pointwise = schedule;', 'let options = CompileOptions { use_schedule_pointwise: schedule, ..Default::default() };')
s = s.replace('for c in 0..6 { accurate(output[row * 6 + c], raw[c] / total, "normalized weight"); }', 'for (c, &weight) in raw.iter().enumerate() { accurate(output[row * 6 + c], weight / total, "normalized weight"); }')
s = s.replace('0.28302041', '0.283_020_4')
p.write_text(s)
p = Path('src/cache.rs')
s = p.read_text()
assert s.count('const CACHE_FORMAT_VERSION: u32 = 5;') == 1
s = s.replace('const CACHE_FORMAT_VERSION: u32 = 5;', '// Version 6 invalidates cancellation-prone Softplus/SoftplusGrad lowerings.\nconst CACHE_FORMAT_VERSION: u32 = 6;')
s += '''
#[cfg(test)]
mod softplus_cache_tests {
    use super::*;
    #[test]
    fn old_softplus_execution_plans_are_invalidated() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[2]);
        let y = graph.softplus(x, 1.0);
        graph.set_outputs(vec![y]);
        let legacy = CachedPlan {
            format_version: 5,
            graph_hash: hash_graph(&graph),
            build_hash: 0,
            plan: crate::compile::compile(&graph),
        };
        let path = std::env::temp_dir().join(format!("meganeura-softplus-v5-{}.ron", std::process::id()));
        std::fs::write(&path, ron::ser::to_string(&legacy).unwrap()).unwrap();
        assert!(load_plan(&graph, &path).unwrap().is_none());
        std::fs::remove_file(path).unwrap();
    }
}
'''
p.write_text(s)
