use serde_json::{Value, json};
use std::{
    error::Error,
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn command(program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    if !output.status.success() {
        return Err(format!("{program}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let path = path.to_str().ok_or("non-UTF8 path")?;
    let output =
        command("sha256sum", &[path]).or_else(|_| command("shasum", &["-a", "256", path]))?;
    Ok(output
        .split_whitespace()
        .next()
        .ok_or("missing hash")?
        .into())
}

pub fn write_record(file: &mut File, record: &Value) -> Result<(), Box<dyn Error>> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    serde_json::to_writer_pretty(&mut *file, record)?;
    file.flush()?;
    Ok(())
}

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub fn host_sample() -> Value {
    json!({
        "unix_ms": unix_ms(),
        "loadavg": std::fs::read_to_string("/proc/loadavg").ok(),
        "cpu_ticks": std::fs::read_to_string("/proc/stat").ok().and_then(|s| s.lines().next().map(str::to_owned)),
        "cpu0_frequency_khz": std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").ok(),
    })
}
