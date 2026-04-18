//! External object interop: share a GPU buffer between two fully
//! independent Blade contexts.
//!
//! The host application (playing the role of a game engine, renderer,
//! or any producer that already has tensor data on the GPU) owns its
//! own `blade_graphics::Context`. Meganeura owns a *separate* context.
//! The two are connected only through the shared underlying memory of
//! an external-memory allocation: the producer exports an
//! `ExternalMemorySource` handle (a Vulkan opaque FD, a DMA-BUF, a
//! Win32 HANDLE, or a host pointer), meganeura re-imports it into its
//! own context, and both see the same bytes with no CPU roundtrip on
//! the input path.
//!
//! This example uses `ExternalMemorySource::HostAllocation` because
//! it's the one variant that lets us populate the memory with a plain
//! CPU memcpy without standing up a staging + compute pipeline on the
//! producer side. The same API works for FD / DMA-BUF / Win32 handles
//! — see [`blade_graphics::ExternalMemorySource`].
//!
//! Platform note: Blade's external-memory implementation is Vulkan
//! only at the moment; the Metal and GLES backends will panic from
//! `unimplemented!()` during the import.

use std::alloc::{Layout, alloc_zeroed, dealloc};

use blade_graphics as bg;
use meganeura::{ExternalSlot, Graph, build_inference_session};

fn main() {
    env_logger::init();

    // Problem: y = relu(x @ W)
    const ROWS: usize = 4;
    const COLS: usize = 6;
    const OUT: usize = 3;
    const INPUT_BYTES: usize = ROWS * COLS * 4;
    // HostAllocation imports have a driver-imposed alignment requirement
    // (minImportedHostPointerAlignment — typically 64 or the page size).
    // A page-sized alignment is always safe.
    const ALIGN: usize = 4096;

    // Allocate page-aligned zeroed host memory. The producer writes
    // into it, both contexts import it by pointer.
    //
    // Safety: the layout is valid (non-zero size, power-of-two align).
    let layout = Layout::from_size_align(INPUT_BYTES, ALIGN).unwrap();
    let host_ptr = unsafe { alloc_zeroed(layout) };
    assert!(!host_ptr.is_null(), "host allocation failed");

    // --- Producer context (fully independent of meganeura) ---
    //
    // In a real app this is owned by the host engine. Here we spin up
    // a standalone Blade context to play the role convincingly.
    let producer = unsafe {
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
    .expect("producer context init failed");
    eprintln!(
        "producer device : {}",
        producer.device_information().device_name
    );

    // Producer imports the host pointer as a GPU buffer. Because the
    // memory type is `HostAllocation`, Blade maps the buffer so that
    // `buffer.data()` returns the same pointer back — we can write into
    // it directly from the CPU side and the producer's shaders would
    // see those writes.
    let producer_buf = producer.create_buffer(bg::BufferDesc {
        name: "producer_input",
        size: INPUT_BYTES as u64,
        memory: bg::Memory::External(bg::ExternalMemorySource::HostAllocation(host_ptr as usize)),
    });

    // "Render" / produce the tensor. In reality this would be whatever
    // GPU work produces the data; here we just fill it from the CPU to
    // keep the example focused on the interop mechanics.
    unsafe {
        let p = producer_buf.data() as *mut f32;
        for i in 0..(ROWS * COLS) {
            *p.add(i) = (i as f32 - 10.0) * 0.25;
        }
    }
    eprintln!(
        "producer wrote {} bytes into the shared allocation",
        INPUT_BYTES
    );

    // --- Meganeura: build a graph and a session on its OWN context ---
    let mut g = Graph::new();
    let x = g.input("x", &[ROWS, COLS]);
    let w = g.parameter("w", &[COLS, OUT]);
    let mm = g.matmul(x, w);
    let y = g.relu(mm);
    g.set_outputs(vec![y]);

    // `build_inference_session` internally calls `Session::new`, which
    // initialises a fresh `blade_graphics::Context` owned by the
    // session — no context sharing with `producer`.
    let mut session = build_inference_session(&g);
    eprintln!(
        "meganeura device: {}",
        session.device_information().device_name
    );

    // Bind the shared memory to the "x" slot. Meganeura's context
    // imports the same host pointer through its own
    // VK_EXT_external_memory_host allocation, so its shaders read from
    // exactly the bytes the producer wrote.
    //
    // In a pure GPU-to-GPU scenario (FD / DMA-BUF / Win32) the
    // producer would call `producer.get_external_buffer_source(buf)`
    // to extract the OS handle and pass that `ExternalMemorySource`
    // here instead. The API shape is identical.
    let required = session
        .slot_size(ExternalSlot::Input("x"))
        .expect("graph input 'x'");
    assert!(INPUT_BYTES >= required);
    let source = bg::ExternalMemorySource::HostAllocation(host_ptr as usize);
    session
        .bind_external_buffer(ExternalSlot::Input("x"), source, INPUT_BYTES as u64)
        .expect("import external buffer");

    // Weights go through the usual upload path. They could equally
    // well be imported via `ExternalSlot::Parameter("w")`.
    let w_data: Vec<f32> = (0..(COLS * OUT)).map(|i| (i as f32 - 8.0) * 0.1).collect();
    session.set_parameter("w", &w_data);

    // Run inference, read the output back to CPU.
    session.step();
    session.wait();
    let y: Vec<f32> = session.read_output(ROWS * OUT);

    println!("y = relu(x @ w)  (shape [{ROWS}, {OUT}]):");
    for row in 0..ROWS {
        let slice = &y[row * OUT..(row + 1) * OUT];
        println!("  {:?}", slice);
    }

    // Cross-check against a CPU reference.
    let x_ref: Vec<f32> = (0..(ROWS * COLS)).map(|i| (i as f32 - 10.0) * 0.25).collect();
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

    // Teardown order matters:
    // 1. Drop the session. Meganeura waits for any outstanding work and
    //    releases its Vulkan import of the shared memory. The
    //    underlying host allocation stays live (producer still holds a
    //    reference via `producer_buf`).
    drop(session);
    // 2. Producer releases its reference, dropping the last import.
    producer.destroy_buffer(producer_buf);
    // 3. Now the host memory has no GPU users — safe to free.
    unsafe {
        dealloc(host_ptr, layout);
    }

    println!(
        "done — input bytes travelled producer-GPU → meganeura-GPU without any CPU roundtrip."
    );
}
