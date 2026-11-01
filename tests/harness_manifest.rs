//! The test harness must not silently lose coverage.
//!
//! `Cargo.toml` sets `autotests = false` and declares its test targets by
//! hand, which keeps the suite to a handful of link jobs. The cost is that a
//! file under `tests/` is dead code until a declared root reaches it through
//! `mod` declarations, and nothing in the build notices when it does not:
//! `outline_optimize.rs`, `profile_windows.rs` and `resnet_correctness.rs`
//! once sat uncompiled that way. So this walks the closure the compiler
//! walks and checks both directions: every file on disk is reached, and
//! every `mod` names a file that exists (a `#[cfg]`-disabled `mod` is never
//! loaded, so the compiler would not say).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[test]
fn every_test_file_is_reached_from_a_declared_target() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo = std::fs::read_to_string(manifest.join("Cargo.toml")).expect("Cargo.toml");
    // With autodiscovery back, every file would get its own target and the
    // orphan check below would pass vacuously.
    assert!(
        cargo.lines().any(|l| l.trim() == "autotests = false"),
        "Cargo.toml no longer sets `autotests = false`; revisit this guard"
    );

    // The `[[test]]` roots: `path` when given, else `tests/<name>.rs`. Every
    // table header ends the block being read; the trailing `[` flushes the
    // last one.
    let mut roots = Vec::new();
    let mut table: Option<(Option<String>, Option<String>)> = None;
    for line in cargo.lines().map(str::trim).chain(["["]) {
        if line.starts_with('[') {
            if let Some((Some(name), path)) = table.take() {
                roots.push(PathBuf::from(path.unwrap_or(format!("tests/{name}.rs"))));
            }
            table = (line == "[[test]]").then_some((None, None));
        } else if let Some((name, path)) = &mut table
            && let Some((key, value)) = line.split_once('=')
        {
            let value = Some(value.trim().trim_matches('"').to_string());
            match key.trim() {
                "name" => *name = value,
                "path" => *path = value,
                _ => {}
            }
        }
    }
    assert!(
        !roots.is_empty(),
        "no [[test]] targets parsed from Cargo.toml"
    );

    // Walk the `mod` closure. A file that cannot be read is a `mod` (or a
    // declared root) naming nothing.
    let mut reached = BTreeSet::new();
    let mut missing = Vec::new();
    let mut queue = roots;
    while let Some(file) = queue.pop() {
        if !reached.insert(file.clone()) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(manifest.join(&file)) else {
            missing.push(file.display().to_string());
            continue;
        };
        let dir = file.parent().unwrap_or(Path::new("")).to_path_buf();
        for line in source.lines() {
            let Some(name) = line
                .trim()
                .strip_prefix("mod ")
                .and_then(|l| l.strip_suffix(';'))
            else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let directory = dir.join(name);
            queue.push(if manifest.join(&directory).is_dir() {
                directory.join("mod.rs")
            } else {
                dir.join(format!("{name}.rs"))
            });
        }
    }

    let orphans: Vec<String> = std::fs::read_dir(manifest.join("tests"))
        .expect("tests/ is readable")
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| path.strip_prefix(manifest).unwrap().to_path_buf())
        .filter(|path| !reached.contains(path))
        .map(|path| path.display().to_string())
        .collect();
    assert!(
        orphans.is_empty() && missing.is_empty(),
        "unreachable test files (add `mod <stem>;` to `smoke` for broad stack contracts, \
         `regression` for focused ones, or give them a [[test]] target): {orphans:?}\n\
         `mod` declarations or targets naming no file: {missing:?}"
    );
}
