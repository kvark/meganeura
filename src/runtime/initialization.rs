//! Temporary initialization diagnostics; not a production profiling interface.

use std::fmt::Debug;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(super) struct Trace {
    session: u64,
    started: Instant,
}

fn write_event(writer: &mut impl Write, event: &serde_json::Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, event).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn completed<E: Debug>(result: Result<bool, E>) -> Result<(), String> {
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err("GPU wait did not complete".into()),
        Err(error) => Err(format!("GPU wait failed: {error:?}")),
    }
}

impl Trace {
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            session: NEXT.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
        }
    }

    pub(super) fn mark(&self, phase: &str, data: serde_json::Value) {
        let event = serde_json::json!({
            "meganeura_initialization": 1,
            "session": self.session,
            "pid": std::process::id(),
            "unix_ns": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            "elapsed_ns": self.started.elapsed().as_nanos(),
            "phase": phase,
            "data": data,
        });
        write_event(&mut io::stderr().lock(), &event)
            .expect("initialization diagnostic could not be flushed; stopping");
    }

    pub(super) fn wait<E: Debug>(&self, phase: &str, result: Result<bool, E>) {
        let result = completed(result);
        self.mark(
            phase,
            serde_json::json!({
                "complete": result.is_ok(),
                "error": result.as_ref().err(),
            }),
        );
        result.unwrap_or_else(|error| {
            panic!("{phase}: {error}; stopping before further initialization")
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Sink {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn event_is_one_flushed_json_record() {
        let mut sink = Sink::default();
        let event = serde_json::json!({"phase": "host_zero.before", "data": {"bytes": 4096}});
        write_event(&mut sink, &event).unwrap();
        assert_eq!(sink.flushes, 1);
        assert_eq!(sink.bytes.last(), Some(&b'\n'));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&sink.bytes).unwrap(),
            event
        );
    }

    #[test]
    fn event_escapes_embedded_newlines() {
        let mut sink = Sink::default();
        write_event(&mut sink, &serde_json::json!({"data": "line1\nline2"})).unwrap();
        assert_eq!(sink.bytes.iter().filter(|&&b| b == b'\n').count(), 1);
    }

    #[test]
    fn write_failure_propagates() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("must not flush after a failed write");
            }
        }
        assert!(write_event(&mut Broken, &serde_json::json!({})).is_err());
    }

    #[test]
    fn flush_failure_propagates() {
        struct BrokenFlush;
        impl Write for BrokenFlush {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }
        assert!(write_event(&mut BrokenFlush, &serde_json::json!({})).is_err());
    }

    #[test]
    fn completed_wait_succeeds() {
        assert_eq!(completed::<&str>(Ok(true)), Ok(()));
    }

    #[test]
    fn incomplete_wait_is_not_success() {
        assert_eq!(
            completed::<&str>(Ok(false)),
            Err("GPU wait did not complete".into())
        );
    }

    #[test]
    fn device_loss_retains_its_identity() {
        assert_eq!(
            completed(Err("DeviceLost")),
            Err("GPU wait failed: \"DeviceLost\"".into())
        );
    }

    #[test]
    fn out_of_memory_retains_its_identity() {
        assert_eq!(
            completed(Err("OutOfMemory")),
            Err("GPU wait failed: \"OutOfMemory\"".into())
        );
    }

    #[test]
    fn failed_wait_never_reaches_the_next_operation() {
        let next = std::sync::atomic::AtomicBool::new(false);
        let result = std::panic::catch_unwind(|| {
            Trace::new().wait("zero_device_local.wait", Err("DeviceLost"));
            next.store(true, Ordering::Relaxed);
        });
        assert!(result.is_err());
        assert!(!next.load(Ordering::Relaxed));
    }
}
