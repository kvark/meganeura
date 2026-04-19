//! Analyze generated shader SPIR-V for coop matmul and other key kernels.
//!
//! Generates the actual WGSL that the runtime would use (after template
//! substitution), compiles to SPIR-V, and reports load/store statistics:
//!   - vec4 vs scalar loads (128-bit vs 32-bit memory transactions)
//!   - cooperative matrix ops
//!   - total instruction count
//!
//! With `--gpu`, also compiles each pipeline on the actual GPU and queries
//! driver-reported statistics (register counts, spill loads/stores, etc.).
//!
//! Usage:
//!   cargo run --release --example analyze_shaders
//!   cargo run --release --example analyze_shaders -- --dump   # also write .spv files
//!   cargo run --release --example analyze_shaders -- --gpu    # query GPU pipeline stats

use std::process::Command;

use meganeura::codegen::{CoopConfig, ShaderGroup, ShaderModule};
use meganeura::compile::ShaderEntry;

fn analyze_spirv(name: &str, module: &naga::Module, dump: bool) {
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = match naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
        .validate(module)
    {
        Ok(info) => info,
        Err(e) => {
            eprintln!("  {}: VALIDATION FAILED: {}", name, e);
            return;
        }
    };

    let mut opts = naga::back::spv::Options::default();
    opts.flags |= naga::back::spv::WriterFlags::DEBUG;
    // Match blade's settings: no workgroup memory zeroinit (would add
    // a thread-0-only OpStore of the entire shared array on every dispatch).
    opts.zero_initialize_workgroup_memory =
        naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None;
    let pipeline_opts = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    let words = match naga::back::spv::write_vec(module, &info, &opts, Some(&pipeline_opts)) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("  {}: SPIR-V WRITE FAILED: {}", name, e);
            return;
        }
    };

    if dump {
        let out_path = format!("{}.spv", name);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&out_path, &bytes).unwrap();
        eprintln!("  wrote {} ({} bytes)", out_path, bytes.len());
    }

    // Disassemble and analyze
    let spv_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let tmp = format!("{}.tmp.spv", name);
    std::fs::write(&tmp, &spv_bytes).unwrap();
    let dis = Command::new("spirv-dis").arg(&tmp).output();
    let _ = std::fs::remove_file(&tmp);

    let disasm = match dis {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => {
            eprintln!("  {}: spirv-dis not available, skipping analysis", name);
            return;
        }
    };

    // Count instruction types
    let mut vec4_loads = 0usize;
    let mut scalar_loads = 0usize;
    let mut coop_ops = 0usize;

    for line in disasm.lines() {
        let trimmed = line.trim();

        // Count loads by type
        if trimmed.contains("OpLoad") {
            if trimmed.contains("v4float") || trimmed.contains("_vec4") {
                vec4_loads += 1;
            } else if trimmed.contains("%float")
                && !trimmed.contains("v2float")
                && !trimmed.contains("v3float")
            {
                scalar_loads += 1;
            }
        }

        // Count cooperative matrix operations
        if trimmed.contains("OpCooperativeMatrix") {
            coop_ops += 1;
        }
    }

    let total_loads = vec4_loads + scalar_loads;
    let vec4_pct = if total_loads > 0 {
        vec4_loads as f64 / total_loads as f64 * 100.0
    } else {
        0.0
    };

    println!(
        "  {:40} {:>4} vec4 / {:>4} scalar loads ({:5.1}% vec4), {} coop ops, {} total instructions",
        name,
        vec4_loads,
        scalar_loads,
        vec4_pct,
        coop_ops,
        disasm.lines().count(),
    );
}

fn analyze_gpu(gpu: &blade_graphics::Context, name: &str, sm: &ShaderModule, entry: &ShaderEntry) {
    use blade_graphics as bg;
    let shader = gpu.create_shader(bg::ShaderDesc {
        source: &sm.source,
        naga_module: Some(sm.module.clone()),
    });
    let layout = meganeura::runtime::shader_data_layout(entry);
    let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
        name: entry.entry_point(),
        data_layouts: &[&layout],
        compute: shader.at(entry.entry_point()),
    });
    let stats = gpu.get_pipeline_statistics(&pipeline);
    if stats.is_empty() {
        println!("  {:40} (no GPU statistics available)", name);
    } else {
        for exec in &stats {
            print!("  {:40}", name);
            for stat in &exec.statistics {
                print!("  {}={}", stat.name, stat.value);
            }
            println!();
        }
    }
    gpu.destroy_compute_pipeline(&mut pipeline);
}

fn analyze(
    name: &str,
    sm: &ShaderModule,
    entry: &ShaderEntry,
    dump: bool,
    gpu: Option<&blade_graphics::Context>,
) {
    analyze_spirv(name, &sm.module, dump);
    if let Some(g) = gpu {
        analyze_gpu(g, name, sm, entry);
    }
}

fn main() {
    let dump = std::env::args().any(|a| a == "--dump");
    let gpu_mode = std::env::args().any(|a| a == "--gpu");

    let gpu = if gpu_mode {
        let dev_id = std::env::var("MEGANEURA_DEVICE_ID")
            .ok()
            .and_then(|s| s.parse().ok());
        let ctx = unsafe {
            blade_graphics::Context::init(blade_graphics::ContextDesc {
                validation: false,
                timing: false,
                capture: false,
                overlay: false,
                device_id: dev_id,
                ..Default::default()
            })
        }
        .expect("failed to initialize GPU context");
        let info = ctx.device_information();
        let caps = ctx.capabilities();
        eprintln!(
            "GPU: {:?} (driver={:?}, software={}) coop_matrix={:?}",
            info.device_name, info.driver_name, info.is_software_emulated, caps.cooperative_matrix
        );
        Some(ctx)
    } else {
        None
    };
    let gpu_ref = gpu.as_ref();

    println!("=== Shader Analysis ===\n");

    // 1. Coop matmul (f16 path, tile=16)
    println!("Cooperative matmul (tile=16, f16 input, f32 accum):");
    {
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
        };
        let cases: &[(&str, ShaderGroup, ShaderEntry)] = &[
            (
                "matmul_coop_16x16_f16",
                ShaderGroup::MatMulCoop,
                ShaderEntry::MatMul,
            ),
            (
                "matmul_coop_bt_16x16_f16",
                ShaderGroup::MatMulCoopBT,
                ShaderEntry::MatMulBT,
            ),
            (
                "matmul_coop_at_16x16_f16",
                ShaderGroup::MatMulCoopAT,
                ShaderEntry::MatMulAT,
            ),
            (
                "matmul_coop_add_16x16_f16",
                ShaderGroup::MatMulCoopAdd,
                ShaderEntry::FusedMatMulAdd,
            ),
        ];
        for (name, group, entry) in cases {
            let sm = meganeura::codegen::generate_coop_module(*group, &config);
            analyze(name, &sm, entry, dump, gpu_ref);
        }
    }

    // 2. Coop matmul (f32 path, tile=8 — Apple Silicon)
    println!("\nCooperative matmul (tile=8, f32 — Apple path):");
    {
        let config = CoopConfig {
            tile_size: 8,
            use_f16_input: false,
        };
        let sm = meganeura::codegen::generate_coop_module(ShaderGroup::MatMulCoop, &config);
        analyze(
            "matmul_coop_8x8_f32",
            &sm,
            &ShaderEntry::MatMul,
            dump,
            gpu_ref,
        );
    }

    // 3. Non-coop matmul (register-tiled)
    println!("\nRegister-tiled matmul (no tensor cores):");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::MatMul);
        analyze(
            "matmul_64x64_register",
            &sm,
            &ShaderEntry::MatMul,
            dump,
            gpu_ref,
        );
    }

    // 4. GEMV kernels
    println!("\nGEMV kernels:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::MatMulGemv);
        analyze("matmul_gemv", &sm, &ShaderEntry::MatMulGemv, dump, gpu_ref);

        let sm_bt = meganeura::codegen::generate_module(ShaderGroup::MatMulGemvBT);
        analyze(
            "matmul_gemv_bt",
            &sm_bt,
            &ShaderEntry::MatMulGemvBT,
            dump,
            gpu_ref,
        );
    }

    // 5. Conv2d GEMM (forward)
    println!("\nConv2d GEMM:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::Conv2dGemm);
        analyze(
            "conv2d_gemm_64x64",
            &sm,
            &ShaderEntry::Conv2dGemm,
            dump,
            gpu_ref,
        );

        let sm_small = meganeura::codegen::generate_module(ShaderGroup::Conv2dGemmSmall);
        analyze(
            "conv2d_gemm_32x32",
            &sm_small,
            &ShaderEntry::Conv2dGemmSmall,
            dump,
            gpu_ref,
        );
    }

    // 6. Conv2d backward
    println!("\nConv2d backward:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::Conv2dGradInputGemm);
        analyze(
            "conv2d_grad_input_gemm",
            &sm,
            &ShaderEntry::Conv2dGradInputGemm,
            dump,
            gpu_ref,
        );

        let sm_wt = meganeura::codegen::generate_module(ShaderGroup::Conv2dGradWeightGemm);
        analyze(
            "conv2d_grad_weight_gemm",
            &sm_wt,
            &ShaderEntry::Conv2dGradWeightGemm,
            dump,
            gpu_ref,
        );
    }

    // 7. Attention
    println!("\nAttention:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::MultiHeadAttn);
        analyze(
            "multi_head_attn",
            &sm,
            &ShaderEntry::MultiHeadAttn,
            dump,
            gpu_ref,
        );
    }

    // 8. Flash Attention
    println!("\nFlash Attention:");
    {
        let sm = meganeura::codegen::generate_flash_attention_module(64);
        analyze(
            "flash_attention_hd64",
            &sm,
            &ShaderEntry::FlashAttention,
            dump,
            gpu_ref,
        );

        let sm_gq = meganeura::codegen::generate_flash_grad_q_module(64);
        analyze(
            "flash_grad_q_hd64",
            &sm_gq,
            &ShaderEntry::FlashGradQ,
            dump,
            gpu_ref,
        );

        let sm_gkv = meganeura::codegen::generate_flash_grad_kv_module(64);
        analyze(
            "flash_grad_kv_hd64",
            &sm_gkv,
            &ShaderEntry::FlashGradKV,
            dump,
            gpu_ref,
        );

        let sm_gk = meganeura::codegen::generate_flash_grad_k_module(64);
        analyze(
            "flash_grad_k_hd64",
            &sm_gk,
            &ShaderEntry::FlashGradK,
            dump,
            gpu_ref,
        );

        let sm_gv = meganeura::codegen::generate_flash_grad_v_module(64);
        analyze(
            "flash_grad_v_hd64",
            &sm_gv,
            &ShaderEntry::FlashGradV,
            dump,
            gpu_ref,
        );
    }

    // 9. Normalization backward
    println!("\nNormalization backward:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::RmsNormGrad);
        analyze(
            "rms_norm_grad_w",
            &sm,
            &ShaderEntry::RmsNormGradW,
            dump,
            gpu_ref,
        );
        analyze(
            "rms_norm_grad_x",
            &sm,
            &ShaderEntry::RmsNormGradX,
            dump,
            gpu_ref,
        );

        let sm_lnorm = meganeura::codegen::generate_module(ShaderGroup::LayerNormGrad);
        analyze(
            "layer_norm_grad_wb",
            &sm_lnorm,
            &ShaderEntry::LayerNormGradWB,
            dump,
            gpu_ref,
        );
        analyze(
            "layer_norm_grad_x",
            &sm_lnorm,
            &ShaderEntry::LayerNormGradX,
            dump,
            gpu_ref,
        );
    }

    // 10. Unary/Binary ops
    println!("\nElementwise ops:");
    {
        let sm_unary = meganeura::codegen::generate_module(ShaderGroup::Unary);
        analyze("relu", &sm_unary, &ShaderEntry::Relu, dump, gpu_ref);
        analyze("silu", &sm_unary, &ShaderEntry::Silu, dump, gpu_ref);
        analyze("gelu", &sm_unary, &ShaderEntry::Gelu, dump, gpu_ref);

        let sm_binary = meganeura::codegen::generate_module(ShaderGroup::Binary);
        analyze("add", &sm_binary, &ShaderEntry::Add, dump, gpu_ref);
        analyze("mul", &sm_binary, &ShaderEntry::Mul, dump, gpu_ref);

        let sm_reduce = meganeura::codegen::generate_module(ShaderGroup::Reduce);
        analyze("sum_all", &sm_reduce, &ShaderEntry::SumAll, dump, gpu_ref);
        analyze("mean_all", &sm_reduce, &ShaderEntry::MeanAll, dump, gpu_ref);
    }

    // 11. Softmax + losses
    println!("\nSoftmax/Losses:");
    {
        let sm = meganeura::codegen::generate_module(ShaderGroup::Softmax);
        analyze("softmax", &sm, &ShaderEntry::Softmax, dump, gpu_ref);

        let sm_ce = meganeura::codegen::generate_module(ShaderGroup::CrossEntropy);
        analyze(
            "cross_entropy",
            &sm_ce,
            &ShaderEntry::CrossEntropyLoss,
            dump,
            gpu_ref,
        );
    }

    println!("\nDone. Use --dump to write .spv files for manual inspection.");
    println!("     Use --gpu to query real GPU pipeline statistics.");
    if dump {
        println!("Disassemble with: spirv-dis <name>.spv | grep -E 'OpLoad|OpStore|Cooperative'");
    }
}
