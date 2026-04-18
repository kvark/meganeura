//! Analyze generated shader SPIR-V for coop matmul and other key kernels.
//!
//! Generates the actual WGSL that the runtime would use (after template
//! substitution), compiles to SPIR-V, and reports load/store statistics:
//!   - vec4 vs scalar loads (128-bit vs 32-bit memory transactions)
//!   - cooperative matrix ops
//!   - total instruction count
//!
//! Usage:
//!   cargo run --release --example analyze_shaders
//!   cargo run --release --example analyze_shaders -- --dump   # also write .spv files

use std::collections::HashMap;
use std::process::Command;

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
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut vec4_loads = 0usize;
    let mut scalar_loads = 0usize;
    let mut vec4_type_defined = false;
    let mut coop_ops = 0usize;

    for line in disasm.lines() {
        let trimmed = line.trim();

        // Track vec4<f32> type definitions
        if trimmed.contains("OpTypeVector") && trimmed.contains("%float") && trimmed.contains("4") {
            vec4_type_defined = true;
        }

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
            *counts.entry("coop_matrix_ops").or_default() += 1;
        }

        // Count key instruction categories
        if trimmed.contains("OpFMul") || trimmed.contains("OpFAdd") {
            *counts.entry("float_arith").or_default() += 1;
        }
        if trimmed.contains("OpStore") {
            *counts.entry("stores").or_default() += 1;
        }
        if trimmed.contains("OpLoad") {
            *counts.entry("loads").or_default() += 1;
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

    let _ = vec4_type_defined; // suppress warning
}

fn main() {
    let dump = std::env::args().any(|a| a == "--dump");

    println!("=== Shader SPIR-V Analysis ===\n");

    // 1. Coop matmul Normal (f16 path, tile=16) — the new vec4 version
    println!("Cooperative matmul (tile=16, f16 input, f32 accum):");
    {
        let config = meganeura::codegen::CoopConfig {
            tile_size: 16,
            use_f16_input: true,
        };
        let sm = meganeura::codegen::generate_coop_module(
            meganeura::codegen::ShaderGroup::MatMulCoop,
            &config,
        );
        analyze_spirv("matmul_coop_16x16_f16", &sm.module, dump);

        let sm_bt = meganeura::codegen::generate_coop_module(
            meganeura::codegen::ShaderGroup::MatMulCoopBT,
            &config,
        );
        analyze_spirv("matmul_coop_bt_16x16_f16", &sm_bt.module, dump);

        let sm_at = meganeura::codegen::generate_coop_module(
            meganeura::codegen::ShaderGroup::MatMulCoopAT,
            &config,
        );
        analyze_spirv("matmul_coop_at_16x16_f16", &sm_at.module, dump);

        let sm_add = meganeura::codegen::generate_coop_module(
            meganeura::codegen::ShaderGroup::MatMulCoopAdd,
            &config,
        );
        analyze_spirv("matmul_coop_add_16x16_f16", &sm_add.module, dump);
    }

    // 2. Coop matmul (f32 path, tile=8 — Apple Silicon)
    println!("\nCooperative matmul (tile=8, f32 — Apple path):");
    {
        let config = meganeura::codegen::CoopConfig {
            tile_size: 8,
            use_f16_input: false,
        };
        let sm = meganeura::codegen::generate_coop_module(
            meganeura::codegen::ShaderGroup::MatMulCoop,
            &config,
        );
        analyze_spirv("matmul_coop_8x8_f32", &sm.module, dump);
    }

    // 3. Non-coop matmul (register-tiled)
    println!("\nRegister-tiled matmul (no tensor cores):");
    {
        let sm = meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::MatMul);
        analyze_spirv("matmul_64x64_register", &sm.module, dump);
    }

    // 4. GEMV kernels
    println!("\nGEMV kernels:");
    {
        let sm = meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::MatMulGemv);
        analyze_spirv("matmul_gemv", &sm.module, dump);

        let sm_bt =
            meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::MatMulGemvBT);
        analyze_spirv("matmul_gemv_bt", &sm_bt.module, dump);
    }

    // 5. Conv2d GEMM (forward)
    println!("\nConv2d GEMM:");
    {
        let sm = meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::Conv2dGemm);
        analyze_spirv("conv2d_gemm_64x64", &sm.module, dump);

        let sm_small =
            meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::Conv2dGemmSmall);
        analyze_spirv("conv2d_gemm_32x32", &sm_small.module, dump);
    }

    // 6. Conv2d backward
    println!("\nConv2d backward:");
    {
        let sm = meganeura::codegen::generate_module(
            meganeura::codegen::ShaderGroup::Conv2dGradInputGemm,
        );
        analyze_spirv("conv2d_grad_input_gemm", &sm.module, dump);

        let sm_wt = meganeura::codegen::generate_module(
            meganeura::codegen::ShaderGroup::Conv2dGradWeightGemm,
        );
        analyze_spirv("conv2d_grad_weight_gemm", &sm_wt.module, dump);
    }

    // 7. Attention
    println!("\nAttention:");
    {
        let sm =
            meganeura::codegen::generate_module(meganeura::codegen::ShaderGroup::CausalAttention);
        analyze_spirv("causal_attention", &sm.module, dump);
    }

    println!("\nDone. Use --dump to write .spv files for manual inspection.");
    if dump {
        println!("Disassemble with: spirv-dis <name>.spv | grep -E 'OpLoad|OpStore|Cooperative'");
    }
}
