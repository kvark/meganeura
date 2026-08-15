//! Print the generated WGSL for one shader group.
//!
//! Usage: cargo run --example print_shader [MultiHeadAttn|FlashAttention|RmsNorm|MatMul]

use meganeura::codegen::{CoopConfig, ShaderGroup, generate_module_coop, generate_wgsl};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let group = args.get(1).map(|s| s.as_str()).unwrap_or("MatMulCoop");
    let wgsl = match group {
        "MultiHeadAttn" => generate_wgsl(ShaderGroup::MultiHeadAttn),
        "FlashAttention" => generate_wgsl(ShaderGroup::FlashAttention),
        "RmsNorm" => generate_wgsl(ShaderGroup::RmsNorm),
        // Cooperative execution is a modifier, not a group of its own, so
        // it needs a tile config rather than a group name.
        _ => {
            let config = CoopConfig {
                tile_size: 16,
                use_f16_input: true,
                compensated: false,
            };
            generate_module_coop(ShaderGroup::MatMul, &config).source
        }
    };
    println!("{}", wgsl);
}
