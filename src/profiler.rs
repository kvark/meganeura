//! Profiling infrastructure producing Perfetto binary traces (`.pftrace`).
//!
//! CPU-side work is captured automatically via [`tracing`] spans. GPU pass
//! durations come from blade-graphics hardware timestamp queries. Both land
//! on separate tracks in the resulting trace, viewable in
//! [Perfetto UI](https://ui.perfetto.dev).
//!
//! # Quick start
//!
//! ```ignore
//! meganeura::profiler::init();          // sets up tracing subscriber
//! // ... build session, train ...
//! meganeura::profiler::save("trace.pftrace").unwrap();
//! ```

use std::{
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use tracing::{span, Subscriber};
use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan, Layer};

// ---- Track IDs ----

const CPU_TRACK_UUID: u64 = 1;
const GPU_TRACK_UUID: u64 = 2;

// ---- Trace event model ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum EventKind {
    SliceBegin = 1,
    SliceEnd = 2,
    Instant = 3,
}

struct TraceEvent {
    name: String,
    timestamp_ns: u64,
    track_uuid: u64,
    kind: EventKind,
}

// ---- Shared profiler state ----

struct ProfilerInner {
    epoch: Instant,
    events: Vec<TraceEvent>,
}

impl ProfilerInner {
    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }
}

static PROFILER: OnceLock<Arc<Mutex<ProfilerInner>>> = OnceLock::new();

fn get_or_init() -> &'static Arc<Mutex<ProfilerInner>> {
    PROFILER.get_or_init(|| {
        Arc::new(Mutex::new(ProfilerInner {
            epoch: Instant::now(),
            events: Vec::with_capacity(8192),
        }))
    })
}

// ---- Public API ----

/// Initialize profiling: installs a global [`tracing`] subscriber that records
/// spans as Perfetto slice events on the CPU track.
///
/// Safe to call multiple times (subsequent calls are no-ops).
/// Must be called *before* any tracing spans you want captured.
pub fn init() {
    let inner = get_or_init().clone();
    let layer = ProfileLayer { inner };
    let subscriber = tracing_subscriber::registry().with(layer);
    // Ignore error if a subscriber is already set.
    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// Record GPU pass timing events on the GPU track.
///
/// `submit_offset_ns` is the nanosecond offset (relative to profiler epoch)
/// when the GPU work was submitted. Pass durations are laid out sequentially
/// starting from that offset.
pub fn record_gpu_passes(submit_offset_ns: u64, passes: &[(String, Duration)]) {
    if let Some(inner) = PROFILER.get() {
        let mut guard = inner.lock().unwrap();
        let mut offset = submit_offset_ns;
        for &(ref name, dur) in passes {
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: offset,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            offset += dur.as_nanos() as u64;
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: offset,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }
}

/// Record a single CPU event (for use outside tracing spans).
pub fn record_instant(name: &str) {
    if let Some(inner) = PROFILER.get() {
        let mut guard = inner.lock().unwrap();
        let ts = guard.now_ns();
        guard.events.push(TraceEvent {
            name: name.to_string(),
            timestamp_ns: ts,
            track_uuid: CPU_TRACK_UUID,
            kind: EventKind::Instant,
        });
    }
}

/// Return the nanosecond offset from the profiler epoch (for GPU timing placement).
pub fn now_ns() -> u64 {
    PROFILER
        .get()
        .map(|inner| inner.lock().unwrap().now_ns())
        .unwrap_or(0)
}

/// Number of recorded events (including both CPU spans and GPU passes).
pub fn event_count() -> usize {
    PROFILER
        .get()
        .map(|inner| inner.lock().unwrap().events.len())
        .unwrap_or(0)
}

/// Write all collected events to a Perfetto `.pftrace` binary trace file.
pub fn save(path: impl AsRef<Path>) -> std::io::Result<()> {
    let inner = PROFILER.get().ok_or_else(|| {
        std::io::Error::other("profiler not initialized")
    })?;
    let guard = inner.lock().unwrap();
    write_pftrace(path.as_ref(), &guard.events)
}

// ---- Tracing Layer ----

/// A [`tracing_subscriber::Layer`] that captures span enter/exit as Perfetto
/// slice events on the CPU track.
pub struct ProfileLayer {
    inner: Arc<Mutex<ProfilerInner>>,
}

impl<S> Layer<S> for ProfileLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            guard.events.push(TraceEvent {
                name: span.name().to_string(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            guard.events.push(TraceEvent {
                name: span.name().to_string(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut guard = self.inner.lock().unwrap();
        let ts = guard.now_ns();
        guard.events.push(TraceEvent {
            name: event.metadata().name().to_string(),
            timestamp_ns: ts,
            track_uuid: CPU_TRACK_UUID,
            kind: EventKind::Instant,
        });
    }
}

// ---- Perfetto binary trace writer ----
//
// Minimal protobuf encoder — just enough to produce valid .pftrace files
// without pulling in prost or other heavy dependencies.

/// Write a Perfetto trace file from collected events.
fn write_pftrace(path: &Path, events: &[TraceEvent]) -> std::io::Result<()> {
    use std::io::Write;
    let mut trace = ProtoBuf::new();

    // Process descriptor packet.
    let mut proc_desc = ProtoBuf::new();
    proc_desc.uint32(1, std::process::id()); // pid
    let mut track_desc = ProtoBuf::new();
    track_desc.uint64(1, 0); // uuid (process track)
    track_desc.message(3, &proc_desc); // process
    track_desc.string(2, "meganeura"); // name
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &track_desc); // track_descriptor
    pkt.uint32(10, 1); // trusted_packet_sequence_id
    trace.message(1, &pkt); // Trace.packet

    // CPU track descriptor.
    let mut td = ProtoBuf::new();
    td.uint64(1, CPU_TRACK_UUID);
    td.uint64(5, 0); // parent_uuid → process
    td.string(2, "CPU");
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &td);
    pkt.uint32(10, 1);
    trace.message(1, &pkt);

    // GPU track descriptor.
    let mut td = ProtoBuf::new();
    td.uint64(1, GPU_TRACK_UUID);
    td.uint64(5, 0); // parent_uuid → process
    td.string(2, "GPU");
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &td);
    pkt.uint32(10, 1);
    trace.message(1, &pkt);

    // Sort events by timestamp so Perfetto sees them in order within the
    // shared packet sequence. GPU pass events are appended after CPU events
    // but carry earlier timestamps (the submit offset), which causes
    // "misplaced End" warnings if written in insertion order.
    let mut sorted: Vec<usize> = (0..events.len()).collect();
    sorted.sort_by_key(|&i| events[i].timestamp_ns);

    // Event packets.
    for &i in &sorted {
        let ev = &events[i];
        let mut te = ProtoBuf::new();
        te.uint64(11, ev.track_uuid); // track_uuid
        te.int32(9, ev.kind as i32); // type enum
        te.string(23, &ev.name); // name

        let mut pkt = ProtoBuf::new();
        pkt.uint64(8, ev.timestamp_ns); // timestamp
        pkt.message(11, &te); // track_event
        pkt.uint32(10, 1); // trusted_packet_sequence_id
        trace.message(1, &pkt);
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&trace.buf)?;
    Ok(())
}

// ---- Minimal protobuf encoder ----

struct ProtoBuf {
    buf: Vec<u8>,
}

impl ProtoBuf {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    fn write_varint(&mut self, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn tag(&mut self, field: u32, wire_type: u32) {
        self.write_varint(((field as u64) << 3) | wire_type as u64);
    }

    fn uint64(&mut self, field: u32, val: u64) {
        self.tag(field, 0);
        self.write_varint(val);
    }

    fn uint32(&mut self, field: u32, val: u32) {
        self.tag(field, 0);
        self.write_varint(val as u64);
    }

    fn int32(&mut self, field: u32, val: i32) {
        self.tag(field, 0);
        // Protobuf int32 uses varint with sign extension to 64 bits.
        self.write_varint(val as u32 as u64);
    }

    fn string(&mut self, field: u32, val: &str) {
        self.tag(field, 2);
        self.write_varint(val.len() as u64);
        self.buf.extend_from_slice(val.as_bytes());
    }

    fn message(&mut self, field: u32, msg: &ProtoBuf) {
        self.tag(field, 2);
        self.write_varint(msg.buf.len() as u64);
        self.buf.extend_from_slice(&msg.buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        let mut pb = ProtoBuf::new();
        pb.write_varint(0);
        assert_eq!(pb.buf, &[0]);

        let mut pb = ProtoBuf::new();
        pb.write_varint(1);
        assert_eq!(pb.buf, &[1]);

        let mut pb = ProtoBuf::new();
        pb.write_varint(300);
        assert_eq!(pb.buf, &[0xAC, 0x02]);
    }

    #[test]
    fn test_save_produces_nonempty_file() {
        // Initialize the profiler for this test.
        let inner = get_or_init();
        {
            let mut guard = inner.lock().unwrap();
            guard.events.push(TraceEvent {
                name: "test_span".into(),
                timestamp_ns: 1000,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            guard.events.push(TraceEvent {
                name: "test_span".into(),
                timestamp_ns: 2000,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
            guard.events.push(TraceEvent {
                name: "matmul".into(),
                timestamp_ns: 1200,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            guard.events.push(TraceEvent {
                name: "matmul".into(),
                timestamp_ns: 1800,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }

        let dir = std::env::temp_dir().join("meganeura_profiler_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.pftrace");
        save(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // Should be a non-trivial protobuf file.
        assert!(bytes.len() > 50, "trace file too small: {} bytes", bytes.len());
        // First byte should be a protobuf tag for field 1, wire type 2 (length-delimited).
        assert_eq!(bytes[0] & 0x07, 2, "expected length-delimited wire type");
        assert_eq!(bytes[0] >> 3, 1, "expected field number 1 (Trace.packet)");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);

        // Clean up events for other tests.
        inner.lock().unwrap().events.clear();
    }

    #[test]
    fn test_record_gpu_passes() {
        let inner = get_or_init();
        inner.lock().unwrap().events.clear();

        record_gpu_passes(
            5000,
            &[
                ("relu".into(), Duration::from_nanos(100)),
                ("matmul".into(), Duration::from_nanos(500)),
            ],
        );

        let guard = inner.lock().unwrap();
        assert_eq!(guard.events.len(), 4); // 2 begin + 2 end
        assert_eq!(guard.events[0].name, "relu");
        assert_eq!(guard.events[0].timestamp_ns, 5000);
        assert_eq!(guard.events[0].kind, EventKind::SliceBegin);
        assert_eq!(guard.events[1].timestamp_ns, 5100); // 5000 + 100
        assert_eq!(guard.events[1].kind, EventKind::SliceEnd);
        assert_eq!(guard.events[2].name, "matmul");
        assert_eq!(guard.events[2].timestamp_ns, 5100);
        assert_eq!(guard.events[3].timestamp_ns, 5600); // 5100 + 500

        drop(guard);
        inner.lock().unwrap().events.clear();
    }

    #[test]
    fn test_now_ns_increases() {
        let _ = get_or_init();
        let t1 = now_ns();
        for _ in 0..1000 {
            std::hint::black_box(0);
        }
        let t2 = now_ns();
        assert!(t2 >= t1);
    }
}
