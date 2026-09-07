use meganeura::{Graph, Mode, SessionConfig, optimize::{self, OptimizeConfig}};

fn graph() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[64 * 8 * 8]);
    let z = g.input("z", &[64 * 8 * 8]);
    // A legitimate user parameter suffix must not be mistaken for cache metadata.
    let w = g.parameter("kernel:winograd", &[64 * 64 * 9]);
    let a = g.conv2d(x, w, 1, 64, 8, 8, 64, 3, 3, 1, 1);
    let b = g.conv2d(z, w, 1, 64, 8, 8, 64, 3, 3, 1, 1);
    g.set_outputs(vec![a,b]);
    g
}

#[test]
fn tied_convolutions_share_one_derived_weight() {
    let mut g=graph();
    optimize::apply_winograd_conv_fusions(&mut g,&mut Vec::new(),&OptimizeConfig::default());
    assert_eq!(g.derived_params.len(),1);
    let nodes:Vec<_>=g.nodes().iter().filter(|n|matches!(n.op,meganeura::graph::Op::WinogradConv2d{..})).collect();
    assert_eq!(nodes.len(),2);
    assert_eq!(nodes[0].inputs[1],nodes[1].inputs[1]);
}

#[test]
fn portable_checkpoint_roundtrips_between_direct_and_shared_winograd() {
    let context=std::sync::Arc::new(meganeura::init_gpu_context_with(Default::default()).unwrap());
    let build=|direct| meganeura::build(&graph(),SessionConfig {
        mode:Mode::Inference,gpu:Some(context.clone()),
        optimize:OptimizeConfig{no_winograd:direct,..Default::default()},..Default::default()
    }).0;
    let mut direct=build(true);
    let mut optimized=build(false);
    let weights:Vec<_>=(0..64*64*9).map(|i|((i*13%29)as f32-14.0)*0.0005).collect();
    let x:Vec<_>=(0..64*8*8).map(|i|(i%31)as f32/31.0).collect();
    let z:Vec<_>=x.iter().map(|v|0.2-v).collect();
    direct.set_parameter("kernel:winograd",&weights);
    let path=std::env::temp_dir().join(format!("shared-winograd-{}.safetensors",std::process::id()));
    direct.save_checkpoint(&path).unwrap();
    optimized.load_checkpoint(&path).unwrap();
    for s in [&mut direct,&mut optimized] {s.set_input("x",&x);s.set_input("z",&z);s.step();s.wait();}
    for i in 0..2 {
        let mut a=vec![0.0;64*8*8];let mut b=a.clone();
        direct.read_output_by_index(i,&mut a);optimized.read_output_by_index(i,&mut b);
        let error=a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        assert!(error<1e-4,"output {i}: {error}");
    }
    optimized.save_checkpoint(&path).unwrap();
    direct.set_parameter("kernel:winograd",&vec![0.0;weights.len()]);
    direct.load_checkpoint(&path).unwrap();
    let mut restored=vec![0.0;weights.len()];direct.read_param("kernel:winograd",&mut restored);
    assert_eq!(restored,weights);
    let bytes=std::fs::read(&path).unwrap();
    let tensors=safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.names(),vec!["kernel:winograd"]);
    std::fs::remove_file(path).unwrap();
}
