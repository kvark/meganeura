fn main() {
    let args: Vec<String> = std::env::args().collect();
    let group = args.get(1).map(|s| s.as_str()).unwrap_or("MatMulCoop");
    let wgsl = match group {
        "MultiHeadAttn" => {
            meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::MultiHeadAttn)
        }
        "FlashAttention" => {
            meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::FlashAttention)
        }
        "RmsNorm" => meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::RmsNorm),
        _ => meganeura::codegen::generate_wgsl(meganeura::codegen::ShaderGroup::MatMulCoop),
    };
    println!("{}", wgsl);
}
