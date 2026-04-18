//! External object interop: share a GPU buffer between a standalone
//! Blade context (playing the role of a host renderer / game engine)
//! and meganeura.
//!
//! The input tensor is allocated and populated *outside* meganeura —
//! directly on a `blade_graphics::Context` the caller owns. Meganeura
//! then consumes that buffer in place, runs a small compute graph, and
//! we read the result back to CPU at the end.
//!
//! This mirrors the intended embedding pattern: the host app already
//! has its data resident on the GPU (from rendering, simulation, a
//! game engine, ...), and we want neural inference to consume it
//! without a CPU roundtrip.

use std::sync::Arc;

use blade_graphics as bg;
use meganeura::{ExternalSlot, Graph, Mode, SessionConfig, build};

fn main() {
    env_logger::init();

    // --- Host application's Blade context ---
    //
    // In a real embedding this is owned by the renderer / game. We
    // wrap it in `Arc` so both sides can share it.
    let gpu = Arc::new(
        unsafe {
            bg::Context::init(bg::ContextDesc {
                validation: cfg!(debug_assertions),
                timing: false,
                capture: false,
                overlay: false,
                device_id: std::env::var("MEGANEURA_DEVICE_ID")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                ..Default::default()
            })
        }
        .expect("failed to initialise blade GPU context"),
    );
    eprintln!(
        "running on {}",
        gpu.device_information().device_name
    );

    // Problem: y = relu(x @ W), where:
    //   x is a [ROWS, COLS] tensor the *host* produced
    //   W is a [COLS, OUT] weight matrix owned by meganeura
    const ROWS: usize = 4;
    const COLS: usize = 6;
    const OUT: usize = 3;

    // --- Step 1: host allocates and fills the input buffer on the GPU ---
    //
    // `Memory::Shared` gives us a host-visible, coherent mapping so we
    // can write into it with a plain pointer. In a real embedding the
    // buffer might instead be written by the host's own compute/blit
    // dispatches, or be device-local and populated via staging.
    let input_bytes = (ROWS * COLS * 4) as u64;
    let external_input = gpu.create_buffer(bg::BufferDesc {
        name: "external_input",
        size: input_bytes,
        memory: bg::Memory::Shared,
    });
    unsafe {
        let p = external_input.data() as *mut f32;
        for i in 0..(ROWS * COLS) {
            *p.add(i) = (i as f32 - 10.0) * 0.25;
        }
    }
    eprintln!(
        "host-owned input buffer: {} bytes, filled from CPU (no meganeura involved)",
        input_bytes
    );

    // --- Step 2: build a meganeura graph that references "x" by name ---
    let mut g = Graph::new();
    let x = g.input("x", &[ROWS, COLS]);
    let w = g.parameter("w", &[COLS, OUT]);
    let mm = g.matmul(x, w);
    let y = g.relu(mm);
    g.set_outputs(vec![y]);

    // --- Step 3: build an inference session on the *shared* context ---
    //
    // `SessionConfig::gpu` routes meganeura to reuse our context
    // instead of initialising its own. That's the precondition for
    // sharing buffers — both sides must agree on device + queue.
    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            gpu: Some(Arc::clone(&gpu)),
            ..SessionConfig::default()
        },
    );

    // --- Step 4: bind the host's buffer to the "x" slot ---
    //
    // After this call, meganeura will read `x` from `external_input`
    // instead of from its own allocation. The internal buffer for "x"
    // is destroyed immediately; `external_input` is *not* destroyed
    // on session drop — we retain ownership.
    let required = session
        .slot_size(ExternalSlot::Input("x"))
        .expect("graph has an input named x");
    assert!(input_bytes as usize >= required);
    session
        .bind_external_buffer(ExternalSlot::Input("x"), external_input)
        .expect("bind external input");

    // Upload weights via the usual path. We could also allocate W
    // externally and bind via `ExternalSlot::Parameter("w")`.
    let w_data: Vec<f32> = (0..(COLS * OUT))
        .map(|i| (i as f32 - 8.0) * 0.1)
        .collect();
    session.set_parameter("w", &w_data);

    // --- Step 5: run meganeura, read the result back to CPU ---
    session.step();
    session.wait();

    let y: Vec<f32> = session.read_output(ROWS * OUT);
    println!("y = relu(x @ w)  (shape [{ROWS}, {OUT}]):");
    for row in 0..ROWS {
        let slice = &y[row * OUT..(row + 1) * OUT];
        println!("  {:?}", slice);
    }

    // --- Step 6: cross-check against a CPU reference ---
    let x_ref: Vec<f32> = (0..(ROWS * COLS))
        .map(|i| (i as f32 - 10.0) * 0.25)
        .collect();
    let mut cpu = [0.0_f32; ROWS * OUT];
    for r in 0..ROWS {
        for o in 0..OUT {
            let mut acc = 0.0_f32;
            for c in 0..COLS {
                acc += x_ref[r * COLS + c] * w_data[c * OUT + o];
            }
            cpu[r * OUT + o] = acc.max(0.0);
        }
    }
    let max_err = y
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("max |gpu - cpu| = {:e}", max_err);
    assert!(max_err < 1e-4, "GPU result diverged from CPU reference");

    // --- Step 7: tear down in the right order ---
    //
    // Drop the session first so any GPU work finishes and meganeura
    // releases its *internal* resources. Then the caller destroys
    // their buffer, and finally the Arc<Context> drops when the last
    // reference goes out of scope.
    drop(session);
    gpu.destroy_buffer(external_input);

    println!("done — buffer was shared GPU-to-GPU, no CPU roundtrip on input.");
}
