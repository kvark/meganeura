//! CPU-only experiment: rustc --edition=2024 -O bench/transpose.rs -o /tmp/transpose-study

#[path = "../src/data/transpose.rs"]
mod transpose;

use std::{hint::black_box, io::Write, time::Instant};

fn main() {
    let source = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(source.status.success());
    let source = String::from_utf8(source.stdout).unwrap();
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .unwrap();
    assert!(
        status.status.success() && status.stdout.is_empty(),
        "commit source first"
    );
    let destination = std::env::args().nth(1).expect("new output CSV path");
    let mut output = std::io::BufWriter::new(std::fs::File::create_new(destination).unwrap());
    writeln!(output, "source,replicate,rows,columns,tile,elapsed_ns").unwrap();
    let shapes = [
        (0, 17),
        (17, 0),
        (1, 33),
        (17, 33),
        (576, 576),
        (1536, 576),
        (576, 1536),
        (2560, 960),
        (960, 2560),
        (8192, 2048),
        (2048, 8192),
        (49152, 2048),
    ];
    let mut checked = 0u64;
    for replicate in 0..4 {
        for (rows, columns) in shapes {
            let data: Vec<_> = (0..rows * columns)
                .map(|i| f32::from_bits((i as u32).wrapping_mul(1664525).wrapping_add(1013904223)))
                .collect();
            let mut tiles = [0, 16, 32, 64];
            tiles.rotate_left(replicate);
            for tile in tiles {
                let start = Instant::now();
                let result = transpose::transpose(black_box(&data), rows, columns, tile);
                let elapsed = start.elapsed().as_nanos();
                for column in 0..columns {
                    for row in 0..rows {
                        assert_eq!(
                            result[column * rows + row].to_bits(),
                            data[row * columns + column].to_bits()
                        );
                    }
                }
                checked += result.len() as u64;
                writeln!(
                    output,
                    "{},{replicate},{rows},{columns},{tile},{elapsed}",
                    source.trim()
                )
                .unwrap();
                output.flush().unwrap();
                println!(
                    "r{} {rows}x{columns} tile={tile}: {:.3} ms",
                    replicate + 1,
                    elapsed as f64 / 1e6
                );
            }
        }
    }
    println!("All {checked} copied elements match bitwise, including non-finite bit patterns.");
}
