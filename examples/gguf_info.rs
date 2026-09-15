//! Inspect a GGUF file: architecture metadata, tensor inventory, and how
//! each tensor would reach a Meganeura parameter.
//!
//! ```text
//! cargo run --example gguf_info -- model.gguf
//! ```
//!
//! The `stored as` column is what matters for weight fidelity. A `DType`
//! means the tensor reaches the GPU losslessly through
//! `Session::set_parameter_packed`. `f16`/`f32` means it is not a packed
//! format and is read with `to_f32`. `unsupported` means this loader does
//! not implement that `ggml_type`, so the tensor is listed but cannot be
//! read.

use std::collections::BTreeMap;
use std::path::PathBuf;

use meganeura::load::gguf::{GgufError, GgufValue, load_gguf};

fn render(value: &GgufValue) -> String {
    match *value {
        GgufValue::String(ref s) if s.len() > 60 => format!("{:.57}...", s),
        GgufValue::String(ref s) => s.clone(),
        GgufValue::Array(ref a) => format!("[{} items]", a.len()),
        GgufValue::Bool(b) => b.to_string(),
        ref v => v
            .as_f64()
            .map(|f| {
                if f.fract() == 0.0 {
                    format!("{f:.0}")
                } else {
                    format!("{f}")
                }
            })
            .unwrap_or_else(|| format!("{v:?}")),
    }
}

fn main() {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: gguf_info <model.gguf>");
        std::process::exit(2);
    };

    let model = match load_gguf(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    println!(
        "architecture: {}",
        model.architecture().unwrap_or("(unset)")
    );
    println!("\nmetadata ({}):", model.metadata.len());
    for (key, value) in model.metadata.iter().collect::<BTreeMap<_, _>>() {
        println!("  {key:<44} {}", render(value));
    }

    println!("\ntensors ({}):", model.tensors.len());
    println!("  {:<44} {:<18} {:<9} stored as", "name", "shape", "type");
    let mut total_bytes = 0usize;
    let mut unsupported = 0usize;
    for (name, t) in model.tensors.iter().collect::<BTreeMap<_, _>>() {
        total_bytes += t.data().len();
        // Metadata only - converting a tensor just to print its
        // destination would allocate the whole payload.
        let stored = match t.packed_dtype() {
            Ok(dtype) => format!("{dtype:?}"),
            // Not a packed format; the GGML type is the host encoding.
            Err(GgufError::UnsupportedPack(ty)) => format!("{ty:?}").to_ascii_lowercase(),
            Err(GgufError::UnsupportedType(tag)) => {
                unsupported += 1;
                format!("unsupported (tag {tag})")
            }
            // Shapes this loader has no parameter form for, e.g. rank > 2.
            Err(e) => format!("unpackable ({e})"),
        };
        println!(
            "  {name:<44} {:<18} {:<9} {stored}",
            format!("{:?}", t.dims),
            format!("{:?}", t.ggml_type),
        );
    }

    println!(
        "\n{:.1} MiB of readable tensor data; {unsupported} tensor(s) use a \
         ggml_type this loader does not implement",
        total_bytes as f64 / (1024.0 * 1024.0),
    );
}
