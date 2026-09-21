use meganeura::{Graph, optimize};
use std::{hint::black_box, time::Instant};

fn main() {
    let mut graph = Graph::new();
    let x = graph.input("x", &[16, 64]);
    let w = graph.parameter("w", &[64, 64]);
    let bias = graph.parameter("bias", &[16, 64]);
    let y = graph.matmul(x, w);
    let y = graph.add(y, bias);
    let y = graph.relu(y);
    graph.set_outputs(vec![y]);
    let program = optimize::dump_egglog_program(&graph);
    let split = program.find("(let $n").unwrap();
    let (prelude, body) = program.split_at(split);
    let mut template = egglog::EGraph::default();
    template.parse_and_run_program(None, prelude).unwrap();
    for mode in ["cold", "clone"] {
        for sample in 0..9 {
            let start = Instant::now();
            for _ in 0..20 {
                let mut egraph = if mode == "clone" { template.clone() } else { egglog::EGraph::default() };
                egraph.parse_and_run_program(None, if mode == "clone" {body} else {&program}).unwrap();
                assert!(egraph.lookup_function(&format!("$n{y}"), &[]).is_some());
                black_box(egraph);
            }
            println!("{}", serde_json::json!({"mode":mode,"sample":sample,"ms":start.elapsed().as_secs_f64()*50.0}));
        }
    }
}
