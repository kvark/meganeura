//! Optional continuous telemetry. Its overhead is part of this experiment.
use super::experiment_io::unix_ms;
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

const MAX_SAMPLES: usize = 40_000;
const FIELDS: &str = "timestamp,uuid,utilization.gpu,memory.used,clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate";

pub struct Monitor {
    child: Option<Child>,
    reader: Option<JoinHandle<()>>,
    samples: Arc<Mutex<Vec<Value>>>,
    error: Option<String>,
}

impl Monitor {
    pub fn start() -> Self {
        let mut monitor = Self {
            child: None,
            reader: None,
            samples: Arc::default(),
            error: None,
        };
        match Command::new("nvidia-smi")
            .args([
                format!("--query-gpu={FIELDS}"),
                "--format=csv,noheader,nounits".into(),
                "--loop-ms=250".into(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                let output = child.stdout.take().unwrap();
                let samples = monitor.samples.clone();
                monitor.reader = Some(std::thread::spawn(move || {
                    for line in BufReader::new(output).lines() {
                        let Ok(line) = line else {
                            break;
                        };
                        let mut samples = samples.lock().unwrap();
                        if samples.len() < MAX_SAMPLES {
                            samples.push(json!({"received_unix_ms": unix_ms(), "csv": line}));
                        }
                    }
                }));
                monitor.child = Some(child);
            }
            Err(error) => monitor.error = Some(error.to_string()),
        }
        monitor
    }

    pub fn finish(mut self) -> Value {
        self.stop();
        let samples = self.samples.lock().unwrap();
        json!({"fields": FIELDS, "requested_interval_ms": 250, "sample_cap": MAX_SAMPLES, "cap_reached": samples.len() == MAX_SAMPLES, "error": self.error, "samples": *samples})
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if let Ok(Some(status)) = child.try_wait() {
                self.error = Some(format!("telemetry exited before stop: {status}"));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop();
    }
}
