//! The test harness must not silently lose coverage.
//!
//! `Cargo.toml` sets `autotests = false` and declares its test targets by
//! hand, which keeps the suite to a handful of link jobs instead of one per
//! file. The cost is that a file under `tests/` is dead code until one of
//! those roots declares it as a `mod` child — and nothing in the build
//! notices when that is missing. A file can be added, never run, and the
//! suite still reports green.
//!
//! That is not hypothetical: `outline_optimize.rs`, `profile_windows.rs`
//! and `resnet_correctness.rs` each carry real assertions and none of them
//! was referenced by any root. So this walks the graph the compiler walks —
//! from the declared targets, through their `mod` declarations — and compares
//! the closure against the files actually on disk.
//!
//! Both directions are checked. A file nothing reaches is dead weight that
//! reads like coverage; a `mod` naming a file that is not there is a missing
//! test that reads as a module. Either is a compile error in any other
//! layout, which is the whole reason this file exists.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Express a path relative to the manifest directory.
///
/// Cargo's `path` entries are already relative (`tests/smoke.rs`) and
/// `read_dir` hands back absolute ones. Comparing the two directly would
/// make every file look unreachable, so both sides are normalised here.
fn relative_to_manifest(path: &Path) -> PathBuf {
    path.strip_prefix(manifest_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| path.to_path_buf())
}

/// Split a `key = value` line, rejecting anything that is not a plain
/// scalar assignment. `required-features = ["gguf"]` is a table, not a
/// scalar, and falls out here rather than being misread as a path.
fn scalar(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }
    Some((key, value.trim().trim_matches('"')))
}

/// The test target roots, read from the `[[test]]` tables in `Cargo.toml`.
///
/// Cargo's default for a target is `tests/<name>.rs` unless it names a
/// `path`, which the oracle target does. Both forms are honoured so this
/// keeps working if a target is renamed or moved.
///
/// Only `[[test]]` blocks count. `Cargo.toml` also declares `[lib]`,
/// `[[bin]]` and around thirty `[[example]]` targets, and every one of those
/// has a `name` — so the table kind has to be tracked rather than assumed.
fn declared_roots() -> Vec<PathBuf> {
    let text = std::fs::read_to_string(manifest_dir().join("Cargo.toml")).expect("Cargo.toml");
    let mut roots = Vec::new();
    let mut in_test_table = false;
    let mut name: Option<String> = None;
    let mut path: Option<String> = None;

    let mut flush = |name: &mut Option<String>, path: &mut Option<String>| {
        let Some(name) = name.take() else {
            path.take();
            return;
        };
        let root = match path.take() {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(format!("tests/{name}.rs")),
        };
        roots.push(root);
    };

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            // Any table header ends the block being read, whether or not it
            // was a `[[test]]`.
            flush(&mut name, &mut path);
            in_test_table = line == "[[test]]";
            continue;
        }
        if !in_test_table {
            continue;
        }
        match scalar(line) {
            Some(("name", value)) => name = Some(value.to_string()),
            Some(("path", value)) => path = Some(value.to_string()),
            _ => {}
        }
    }
    flush(&mut name, &mut path);
    roots
}

/// Resolve bare `mod name;` declarations to sibling files or directory modules.
fn submodules(file: &Path, source: &str) -> Vec<PathBuf> {
    let dir = file.parent().unwrap_or(Path::new(".")).to_path_buf();
    source
        .lines()
        .filter_map(|line| {
            let name = line.trim().strip_prefix("mod ")?.strip_suffix(';')?.trim();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return None;
            }
            let directory = dir.join(name);
            if directory.is_dir() {
                Some(directory.join("mod.rs"))
            } else {
                Some(dir.join(format!("{name}.rs")))
            }
        })
        .collect()
}

/// Walk the `mod` closure from `root`.
///
/// Returns the files that were reached and the ones a `mod` pointed at but
/// that could not be read. A missing file is not `reached`, so it also shows
/// up as a file that is not on disk — that is how it stays visible even if
/// this module's second test is ignored.
fn walk(root: &Path) -> (BTreeSet<PathBuf>, BTreeSet<PathBuf>) {
    let mut reached = BTreeSet::new();
    let mut missing = BTreeSet::new();
    let mut queue = vec![root.to_path_buf()];
    while let Some(file) = queue.pop() {
        if !reached.insert(file.clone()) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(manifest_dir().join(&file)) else {
            missing.insert(file);
            continue;
        };
        queue.extend(submodules(&file, &source));
    }
    (reached, missing)
}

/// Every `.rs` file directly under `tests/`, as manifest-relative paths.
fn test_files_on_disk() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(manifest_dir().join("tests"))
        .expect("tests/ is readable")
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| relative_to_manifest(&path))
        .collect();
    files.sort();
    files
}

#[test]
fn auto_test_discovery_is_still_off() {
    // The guard below reasons from a hand-written target list. If
    // autodiscovery came back, every file would get its own target and the
    // orphan check would pass vacuously — so assert the premise it rests on.
    let text = std::fs::read_to_string(manifest_dir().join("Cargo.toml")).expect("Cargo.toml");
    assert!(
        text.lines().any(|l| l.trim() == "autotests = false"),
        "Cargo.toml no longer sets `autotests = false`, so this guard's premise \
         has changed and `every_test_file_is_reached` needs revisiting"
    );
}

#[test]
fn every_test_file_is_reached_from_a_declared_target() {
    let roots = declared_roots();
    assert!(
        !roots.is_empty(),
        "no [[test]] targets parsed from Cargo.toml; the guard is not looking at anything"
    );

    let mut reached = BTreeSet::new();
    for root in &roots {
        assert!(
            manifest_dir().join(root).exists(),
            "{} is a declared target but not on disk",
            root.display()
        );
        reached.extend(walk(root).0);
    }

    let orphans: Vec<String> = test_files_on_disk()
        .iter()
        .filter(|file| !reached.contains(*file))
        .map(|file| {
            let name = file.file_name().unwrap().to_string_lossy().into_owned();
            let stem = name.trim_end_matches(".rs");
            format!(
                "tests/{name} is not reachable from any [[test]] target, so it is \
                 never compiled and never runs. Add `mod {stem};` to a root — \
                 `smoke` for broad stack contracts, `regression` for focused ones — \
                 or give it its own [[test]] target."
            )
        })
        .collect();

    assert!(
        orphans.is_empty(),
        "{} test file(s) are dead weight:\n{}",
        orphans.len(),
        orphans.join("\n")
    );
}

#[test]
fn every_declared_module_points_at_a_file_that_exists() {
    let mut missing = BTreeSet::new();
    for root in declared_roots() {
        missing.extend(walk(&root).1);
    }

    let listed: Vec<String> = missing
        .iter()
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let stem = name.trim_end_matches(".rs");
            let owner = path
                .parent()
                .map(|dir| dir.display().to_string())
                .unwrap_or_default();
            format!("{owner} declares `mod {stem};` but tests/{name} does not exist")
        })
        .collect();

    assert!(
        listed.is_empty(),
        "{} module declaration(s) resolve to nothing:\n{}",
        listed.len(),
        listed.join("\n")
    );
}
