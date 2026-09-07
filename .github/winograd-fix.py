from pathlib import Path
p=Path('src/optimize.rs'); s=p.read_text(); pos=s.index('pub fn apply_winograd_conv_fusions')
a=s[:pos]; b=s[pos:]
b=b.replace('    let node_ids: Vec<usize>', '    let mut transformed = HashMap::new();\n    let node_ids: Vec<usize>',1)
start=b.index('        // Create Winograd weight parameter name'); end=b.index('        // Rewrite Conv2d',start)
block=b[start:end].replace('        // Create Winograd weight parameter name','        // One transform per logical weight and channel layout.')
block=block.replace('        let wino_param = graph.add_raw_node_with_precision(', '        let parameter = graph.add_raw_node_with_precision(')
block += '        transformed.insert(key, parameter);\n        parameter\n'
new='''        let key = (weight_id, in_channels, out_channels);
        let wino_param = if let Some(&parameter) = transformed.get(&key) {
            graph.nodes_mut()[parameter as usize].requires_full_precision |= requires_full_precision;
            parameter
        } else {
'''+block+'''        };

'''
b=b[:start]+new+b[end:];p.write_text(a+b)
p=Path('src/runtime/checkpoint.rs');s=p.read_text()
old='    let mut seen = HashSet::new();\n    plan.param_buffers\n        .iter()\n        .map'
assert s.count(old)==1
s=s.replace(old,'''    let mut seen = HashSet::new();
    // Winograd execution caches are regenerated from retained logical weights
    // on every forward pass. Other derived/packed weights may be authoritative;
    // never filter those, or ordinary parameters, by a naming convention.
    let caches: HashSet<_> = plan.derived_params.iter().filter_map(|(buffer, _, transform)| {
        matches!(transform, crate::graph::ParamTransform::Winograd3x3 { .. }).then_some(*buffer)
    }).collect();
    plan.param_buffers
        .iter()
        .filter(|(_, buffer)| !caches.contains(buffer))
        .map''',1)
p.write_text(s)
