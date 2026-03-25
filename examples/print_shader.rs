fn main() {
    let args: Vec<String> = std::env::args().collect();
    let group = args.get(1).map(|s| s.as_str()).unwrap_or("MatMulCoop");
    let wgsl = match group {
        "CausalAttention" => {
            meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::CausalAttention)
        }
        "CrossAttention" => {
            meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::CrossAttention)
        }
        "RmsNorm" => meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::RmsNorm),
        _ => meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::MatMulCoop),
    };
    println!("{}", wgsl);
}
