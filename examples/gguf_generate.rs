//! Generate text from a GGUF file and nothing else.
//!
//! ```text
//! cargo run --release --example gguf_generate -- model.gguf "The meaning of life is"
//! cargo run --release --example gguf_generate -- model.gguf "Hello" --tokens 64 --temperature 0.8
//! ```
//!
//! No architecture is named and no dimensions appear below: everything —
//! the graph, the weights, the tokenizer, the special-token ids — is read
//! out of the file. Compare `examples/smollm2.rs`, which hard-codes a
//! config, fetches `tokenizer.json` from the Hub, and spends ~120 lines
//! transposing weights and driving the decode loop by hand.

use std::path::PathBuf;
use std::time::Instant;

use meganeura::load::gguf::{GenerationOptions, GeneratorOptions, load_gguf};

struct Args {
    path: PathBuf,
    prompt: String,
    options: GenerationOptions,
    context: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let path = raw
        .next()
        .ok_or("usage: gguf_generate <model.gguf> [prompt] [flags]")?;
    let mut args = Args {
        path: PathBuf::from(path),
        prompt: String::new(),
        options: GenerationOptions::default(),
        context: 2048,
    };

    let mut rest: Vec<String> = raw.collect();
    let mut positional = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        let take = |i: &mut usize, rest: &mut Vec<String>| -> Result<String, String> {
            *i += 1;
            rest.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", rest[*i - 1]))
        };
        match rest[i].as_str() {
            "--tokens" => args.options.max_tokens = take(&mut i, &mut rest)?.parse().map_err(s)?,
            "--temperature" => {
                args.options.temperature = take(&mut i, &mut rest)?.parse().map_err(s)?
            }
            "--top-k" => args.options.top_k = take(&mut i, &mut rest)?.parse().map_err(s)?,
            "--top-p" => args.options.top_p = take(&mut i, &mut rest)?.parse().map_err(s)?,
            "--seed" => args.options.seed = take(&mut i, &mut rest)?.parse().map_err(s)?,
            "--repeat-penalty" => {
                args.options.repeat_penalty = take(&mut i, &mut rest)?.parse().map_err(s)?
            }
            "--context" => args.context = take(&mut i, &mut rest)?.parse().map_err(s)?,
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    args.prompt = if positional.is_empty() {
        "The meaning of life is".to_string()
    } else {
        positional.join(" ")
    };
    Ok(args)
}

fn s<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn main() {
    env_logger::init();
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let started = Instant::now();
    let model = match load_gguf(&args.path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("could not read {}: {e}", args.path.display());
            std::process::exit(1);
        }
    };
    println!(
        "read {} ({} tensors, {} metadata keys) in {:.2}s",
        args.path.display(),
        model.tensors.len(),
        model.metadata.len(),
        started.elapsed().as_secs_f32(),
    );

    // Everything about the model comes from here. Nothing below names an
    // architecture or a dimension.
    let compiling = Instant::now();
    let mut generator = match meganeura::load::gguf::Generator::with_options(
        &model,
        &GeneratorOptions {
            max_seq_len: args.context,
            ..GeneratorOptions::default()
        },
    ) {
        Ok(generator) => generator,
        Err(e) => {
            eprintln!("could not build a model from this file: {e}");
            std::process::exit(1);
        }
    };

    let config = generator.config();
    println!(
        "{}: {} layers, {} hidden, {}/{} heads of {}, ffn {}, vocab {}",
        config.architecture,
        config.num_layers,
        config.hidden_size,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
    );
    let report = generator.load_report();
    println!(
        "compiled and loaded in {:.2}s: {} tensors kept their block encoding, {} \
         went through f32",
        compiling.elapsed().as_secs_f32(),
        report.packed,
        report.dequantized,
    );
    match generator.vocab() {
        Some(vocab) => println!("tokenizer: {:?}, {} tokens", vocab.kind(), vocab.len()),
        None => {
            eprintln!("this file carries no tokenizer, so it cannot generate from a prompt");
            std::process::exit(1);
        }
    }

    println!("\n{}", args.prompt);
    // Where the sequence stood before this run, so the rate below counts
    // tokens rather than callbacks: a character split across two tokens
    // arrives in one call, and a token held back for a continuation
    // arrives in none.
    let started_at = generator.position();
    let generating = Instant::now();
    let result = generator.generate_streaming(&args.prompt, &args.options, |piece| {
        print!("{piece}");
        // Streaming output is only streaming if it leaves the buffer.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        true
    });
    println!();

    if let Err(e) = result {
        eprintln!("\ngeneration failed: {e}");
        std::process::exit(1);
    }
    let elapsed = generating.elapsed().as_secs_f32();
    // The prompt went through the same sessions, so it counts toward the
    // rate exactly as the generated tokens do.
    let tokens = generator.position().saturating_sub(started_at);
    println!(
        "\n{tokens} tokens in {elapsed:.2}s ({:.1} tok/s), {} of {} context used",
        tokens as f32 / elapsed.max(f32::EPSILON),
        generator.position(),
        generator.max_seq_len(),
    );
}
