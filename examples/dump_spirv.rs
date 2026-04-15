//! Dump the SPIR-V emitted by naga for a given WGSL shader file.
//!
//! Usage:
//!   cargo run --release --example dump_spirv -- src/shaders/matmul_gemv.wgsl
//!
//! Writes `<path>.spv` next to the source. If `spirv-dis` is on PATH (ships
//! with the Vulkan SDK), also prints the disassembly inline. Filter for
//! `OpLoad` instructions to verify vec4 loads aren't being scalarized:
//!
//!   cargo run ... | grep -E "OpLoad|OpType.*Vector"
//!
//! A healthy vec4 load of an f32 array looks like:
//!   %v4f32 = OpTypeVector %float 4
//!   %ptr = OpAccessChain ... %_ptr_StorageBuffer_v4float ...
//!   %val = OpLoad %v4float %ptr
//!
//! A scalarized load looks like a sequence of 4 scalar OpLoads of `%float`,
//! each followed by OpAccessChain with consecutive indices — which would
//! indicate naga is dropping vec4 semantics on storage-buffer loads.

use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: dump_spirv <shader.wgsl>");
    let entry = args.get(1).map(String::as_str).unwrap_or("main");

    let source = std::fs::read_to_string(path).expect("read shader");
    let module = naga::front::wgsl::parse_str(&source).expect("parse WGSL");

    // Validate without binding annotations (meganeura convention — blade
    // fills them at pipeline creation).
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
        .validate(&module)
        .expect("validate");

    let mut opts = naga::back::spv::Options::default();
    opts.flags |= naga::back::spv::WriterFlags::DEBUG; // emit OpName / OpLine
    let pipeline_opts = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: entry.to_string(),
    };
    let words = naga::back::spv::write_vec(&module, &info, &opts, Some(&pipeline_opts))
        .expect("write SPIR-V");

    let out_path = format!("{}.spv", path);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out_path, &bytes).expect("write .spv");
    eprintln!(
        "wrote {} ({} bytes, {} words)",
        out_path,
        bytes.len(),
        words.len()
    );

    // Try to invoke spirv-dis for a readable disassembly on stdout.
    let dis = Command::new("spirv-dis").arg(&out_path).output();
    match dis {
        Ok(out) if out.status.success() => {
            std::io::Write::write_all(&mut std::io::stdout(), &out.stdout).ok();
        }
        Ok(out) => {
            eprintln!(
                "spirv-dis exited with {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            eprintln!(
                "couldn't run spirv-dis ({}). Inspect manually: spirv-dis {}",
                e, out_path
            );
        }
    }
}
