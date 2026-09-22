fn main() {
    let independent = std::env::args().any(|arg| arg == "--independent");
    let mut template = egglog::EGraph::default();
    template.parse_and_run_program(None, "(datatype Op (Leaf i64) (Neg Op))\n(rewrite (Neg (Neg x)) x)").unwrap();
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let template = &template;
            scope.spawn(move || {
                let mut local;
                let template = if independent {
                    local = egglog::EGraph::default();
                    local.parse_and_run_program(None, "(datatype Op (Leaf i64) (Neg Op))\n(rewrite (Neg (Neg x)) x)").unwrap();
                    &local
                } else {
                    template
                };
                for iteration in 0..100 {
                    let mut egraph = template.clone();
                    let program = format!("(let $n0 (Leaf {}))\n(let $n1 (Neg $n0))\n(let $n2 (Neg $n1))\n(run 4)\n(check (= $n2 $n0))", worker * 100 + iteration);
                    egraph.parse_and_run_program(None, &program).unwrap();
                }
            });
        }
    });
    println!("800 concurrent cloned rule databases passed");
}
