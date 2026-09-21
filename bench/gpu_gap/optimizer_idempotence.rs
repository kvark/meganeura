use meganeura::{Graph, graph::Op, optimize};

fn main() {
    for transpose in 0..3 {
        let (m, n, k) = (4, 8, 1024);
        let mut graph = Graph::new();
        let a = graph.input("a", &if transpose == 1 { [k, m] } else { [m, k] });
        let b = graph.parameter("b", &if transpose == 2 { [n, k] } else { [k, n] });
        let c = graph.input("c", &[m, n]);
        let product = match transpose {
            0 => graph.matmul(a, b),
            1 => graph.matmul_at(a, b),
            _ => graph.matmul_bt(a, b),
        };
        let output = graph.add(product, c);
        graph.set_outputs(vec![output]);
        for iteration in 0..3 {
            graph = optimize::optimize_with_report(&graph).0;
            let root = &graph.node(graph.outputs()[0]).op;
            let fused = matches!(root, Op::FusedMatMulAdd | Op::FusedMatMulATAdd | Op::FusedMatMulBTAdd);
            println!("transpose={transpose} iteration={iteration} fused={fused} root={root:?}");
        }
    }
}
