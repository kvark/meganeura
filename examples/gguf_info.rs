//! Inspect a GGUF file: architecture metadata, tensor inventory, and how
//! each tensor would reach a Meganeura parameter.
//!
//! ```text
//! cargo run --example gguf_info -- model.gguf
//! ```
//!
//! The `packs` column is what matters for weight fidelity. `yes` means the
//! tensor repacks losslessly into Meganeura's layout via
//! `Session::set_parameter_packed`. `f32` means it has to be dequantized and
//! requantized through `Session::set_parameter`, which loses precision.

use std::collections::BTreeMap;
use std::path::PathBuf;

use meganeura::load::gguf::{GgufValue, load_gguf};

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
    println!("  {:<44} {:<18} {:<7} packs", "name", "shape", "type");
    let mut total_bytes = 0usize;
    let mut needs_requantize = 0usize;
    for (name, t) in model.tensors.iter().collect::<BTreeMap<_, _>>() {
        total_bytes += t.data.len();
        let packs = match t.to_packed() {
            Ok((dtype, _)) => format!("{dtype:?}"),
            Err(_) => {
                needs_requantize += 1;
                "f32".to_string()
            }
        };
        println!(
            "  {name:<44} {:<18} {:<7} {packs}",
            format!("{:?}", t.dims),
            format!("{:?}", t.ggml_type),
        );
    }

    println!(
        "\n{:.1} MiB of tensor data; {needs_requantize} tensor(s) would be \
         requantized through f32",
        total_bytes as f64 / (1024.0 * 1024.0),
    );
}
