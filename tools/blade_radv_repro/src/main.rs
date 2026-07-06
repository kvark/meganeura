//! Pure-blade repro for the SpectroStream Block(5) zero-output bug.
//!
//! Chain N copies of:
//!   conv-T (Conv2dGradInputHW shader, dispatches [61, 4, 128] workgroups,
//!           writes 7.6M f32 ≈ 30 MiB output)
//!   slice2d (crops back to 3.6M f32 ≈ 14 MiB so next iter can chain)
//!
//! On RADV STRIX1 (AMD Radeon 890M / Phoenix), the chain produces
//! all-zero output once N ≥ 6 when recorded into a single submit.
//! Splitting into one-submit-per-group with vkWaitForFences works.

use blade_graphics as gpu;
use gpu::ShaderData as _;

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct ConvParams {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    stride_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct SliceParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    start_h: u32,
    end_h: u32,
    start_w: u32,
    end_w: u32,
}

#[derive(blade_macros::ShaderData)]
struct ConvData {
    grad_out: gpu::BufferPiece,
    weight: gpu::BufferPiece,
    dst: gpu::BufferPiece,
    params: gpu::BufferPiece,
}

#[derive(blade_macros::ShaderData)]
struct SliceData {
    slice_src: gpu::BufferPiece,
    slice_dst: gpu::BufferPiece,
    slice_params: gpu::BufferPiece,
}

fn main() {
    env_logger::init();
    let ctx = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: false,
            ..Default::default()
        })
        .expect("init")
    };
    println!("Device: {}", ctx.device_information().device_name);

    // SHADER_V env var selects between V1 (original) and V2 (new) shader.
    let source = if std::env::var("SHADER_V").map(|v| v == "1").unwrap_or(false) {
        include_str!("shaders_v1.wgsl")
    } else {
        include_str!("shaders.wgsl")
    };
    let shader = ctx.create_shader(gpu::ShaderDesc {
        source,
        naga_module: None,
    });
    let mut conv_pipe = ctx.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "conv_t",
        data_layouts: &[&ConvData::layout()],
        compute: shader.at("conv_t"),
    });
    let mut slice_pipe = ctx.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "slice",
        data_layouts: &[&SliceData::layout()],
        compute: shader.at("slice"),
    });

    // Shapes (matching SpectroStream d67).
    let batch = 1u32;
    let in_c = 128u32;
    let in_h = 60u32;
    let in_w = 480u32;
    let out_c = 128u32;
    let kh = 3u32;
    let kw = 4u32;
    let stride_h = 1u32;
    let stride_w = 2u32;
    let pad_h = 0u32;
    let pad_w = 0u32;
    let out_h = (in_h - 1) * stride_h + kh - 2 * pad_h; // 62
    let out_w = (in_w - 1) * stride_w + kw - 2 * pad_w; // 962

    // Conv2dGradInputHW's "view":
    //   in_*  = forward conv's input = our ConvT output = [62, 962]
    //   out_* = forward conv's out  = our ConvT input  = [60, 480]
    let conv_in_h = out_h;
    let conv_in_w = out_w;
    let conv_out_h = in_h;
    let conv_out_w = in_w;
    let conv_in_c = out_c;
    let conv_out_c = in_c;

    let small_size = (batch * in_c * in_h * in_w) as usize; // 3.6M
    let big_size = (batch * out_c * out_h * out_w) as usize; // 7.6M
    println!("small={small_size} big={big_size} (each conv-T writes {big_size} elems = {} MiB)",
        big_size * 4 / 1024 / 1024);

    let n_chain = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);
    println!("Chaining {n_chain} conv-T + slice ops");

    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let modes: &[&str] = match mode.as_str() {
        "single" => &["single"],
        "multipass" => &["multipass"],
        "split" => &["split"],
        _ => &["split", "multipass", "single"],
    };

    // Build conv params buffer
    let conv_params = ConvParams {
        batch,
        in_channels: conv_in_c,
        in_h: conv_in_h,
        in_w: conv_in_w,
        out_channels: conv_out_c,
        kernel_h: kh,
        kernel_w: kw,
        stride_h,
        padding_h: pad_h,
        out_h: conv_out_h,
        out_w: conv_out_w,
        padding_w: pad_w,
        stride_w,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let conv_params_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "conv_params",
        size: std::mem::size_of::<ConvParams>() as u64,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(
            &conv_params as *const ConvParams as *const u8,
            conv_params_buf.data(),
            std::mem::size_of::<ConvParams>(),
        );
    }

    // Slice params buffer (crop [62, 962] → [60, 480] using start_h=1 end_h=1 start_w=1 end_w=481)
    let slice_params = SliceParams {
        batch,
        channels: in_c,
        in_h: out_h,
        in_w: out_w,
        start_h: 1,
        end_h: 1,
        start_w: 1,
        end_w: 481,
    };
    let slice_params_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "slice_params",
        size: std::mem::size_of::<SliceParams>() as u64,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(
            &slice_params as *const SliceParams as *const u8,
            slice_params_buf.data(),
            std::mem::size_of::<SliceParams>(),
        );
    }

    // Weights buffer (shared across all conv-T instances; static after upload).
    let weight_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "weight",
        size: (conv_out_c * conv_in_c * kh * kw * 4) as u64,
        memory: gpu::Memory::Shared,
    });
    let weight_data: Vec<f32> = (0..(conv_out_c * conv_in_c * kh * kw))
        .map(|i| (i as f32 * 0.01).cos() * 0.05)
        .collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            weight_data.as_ptr() as *const u8,
            weight_buf.data(),
            weight_data.len() * 4,
        );
    }

    // Allocate alternating "small" (3.6M) and "big" (7.6M) buffers for the chain.
    // n_chain conv-T + n_chain slice ops, sharing 2*n_chain + 1 small buffers and n_chain big buffers.
    let mut small_bufs: Vec<gpu::Buffer> = Vec::with_capacity(n_chain + 1);
    let mut big_bufs: Vec<gpu::Buffer> = Vec::with_capacity(n_chain);
    for i in 0..(n_chain + 1) {
        let memory = if i == 0 || i == n_chain {
            gpu::Memory::Shared // input and final output must be host-visible
        } else {
            gpu::Memory::Device
        };
        small_bufs.push(ctx.create_buffer(gpu::BufferDesc {
            name: "small",
            size: (small_size * 4) as u64,
            memory,
        }));
    }
    for _ in 0..n_chain {
        big_bufs.push(ctx.create_buffer(gpu::BufferDesc {
            name: "big",
            size: (big_size * 4) as u64,
            memory: gpu::Memory::Device,
        }));
    }

    // Init first small buf with non-zero data.
    let init_data: Vec<f32> = (0..small_size).map(|i| (i as f32 * 0.001).sin() + 1.0).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            init_data.as_ptr() as *const u8,
            small_bufs[0].data(),
            init_data.len() * 4,
        );
    }

    let conv_dispatch_wgs = [conv_in_w.div_ceil(16), conv_in_h.div_ceil(16), batch * conv_in_c];
    let slice_total = small_size as u32;
    let slice_dispatch_wgs = [slice_total.div_ceil(256), 1, 1];

    println!("conv-T wgs={:?}, slice wgs={:?}", conv_dispatch_wgs, slice_dispatch_wgs);

    let mut encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
        name: "repro",
        buffer_count: 1,
    });

    let run = |encoder: &mut gpu::CommandEncoder, ctx: &gpu::Context, mode: &str| {
        let record_chain = |encoder: &mut gpu::CommandEncoder, conv_pipe: &gpu::ComputePipeline, slice_pipe: &gpu::ComputePipeline| {
            for i in 0..n_chain {
                // conv-T: small_bufs[i] → big_bufs[i]
                {
                    let mut pass = encoder.compute("conv_t");
                    let mut pc = pass.with(conv_pipe);
                    pc.bind(
                        0,
                        &ConvData {
                            grad_out: small_bufs[i].into(),
                            weight: weight_buf.into(),
                            dst: big_bufs[i].into(),
                            params: conv_params_buf.into(),
                        },
                    );
                    pc.dispatch(conv_dispatch_wgs);
                }
                // slice: big_bufs[i] → small_bufs[i+1]
                {
                    let mut pass = encoder.compute("slice");
                    let mut pc = pass.with(slice_pipe);
                    pc.bind(
                        0,
                        &SliceData {
                            slice_src: big_bufs[i].into(),
                            slice_dst: small_bufs[i + 1].into(),
                            slice_params: slice_params_buf.into(),
                        },
                    );
                    pc.dispatch(slice_dispatch_wgs);
                }
            }
        };

        match mode {
            "single" => {
                encoder.start();
                {
                    let mut pass = encoder.compute("step");
                    for i in 0..n_chain {
                        if i > 0 {
                            pass.barrier();
                        }
                        let mut pc = pass.with(&conv_pipe);
                        pc.bind(
                            0,
                            &ConvData {
                                grad_out: small_bufs[i].into(),
                                weight: weight_buf.into(),
                                dst: big_bufs[i].into(),
                                params: conv_params_buf.into(),
                            },
                        );
                        pc.dispatch(conv_dispatch_wgs);
                        pass.barrier();
                        let mut pc = pass.with(&slice_pipe);
                        pc.bind(
                            0,
                            &SliceData {
                                slice_src: big_bufs[i].into(),
                                slice_dst: small_bufs[i + 1].into(),
                                slice_params: slice_params_buf.into(),
                            },
                        );
                        pc.dispatch(slice_dispatch_wgs);
                    }
                }
                let sp = ctx.submit(encoder);
                let _ = ctx.wait_for(&sp, !0);
            }
            "multipass" => {
                encoder.start();
                record_chain(encoder, &conv_pipe, &slice_pipe);
                let sp = ctx.submit(encoder);
                let _ = ctx.wait_for(&sp, !0);
            }
            "split" => {
                for i in 0..n_chain {
                    encoder.start();
                    {
                        let mut pass = encoder.compute("conv_t");
                        let mut pc = pass.with(&conv_pipe);
                        pc.bind(
                            0,
                            &ConvData {
                                grad_out: small_bufs[i].into(),
                                weight: weight_buf.into(),
                                dst: big_bufs[i].into(),
                                params: conv_params_buf.into(),
                            },
                        );
                        pc.dispatch(conv_dispatch_wgs);
                    }
                    let sp = ctx.submit(encoder);
                    let _ = ctx.wait_for(&sp, !0);

                    encoder.start();
                    {
                        let mut pass = encoder.compute("slice");
                        let mut pc = pass.with(&slice_pipe);
                        pc.bind(
                            0,
                            &SliceData {
                                slice_src: big_bufs[i].into(),
                                slice_dst: small_bufs[i + 1].into(),
                                slice_params: slice_params_buf.into(),
                            },
                        );
                        pc.dispatch(slice_dispatch_wgs);
                    }
                    let sp = ctx.submit(encoder);
                    let _ = ctx.wait_for(&sp, !0);
                }
            }
            _ => panic!("unknown mode {mode}"),
        }
    };

    println!("--- results ---");
    let mut results = Vec::new();
    for &m in modes {
        let t0 = std::time::Instant::now();
        run(&mut encoder, &ctx, m);
        let elapsed = t0.elapsed();
        eprintln!("  mode={m:9} wall time {:.3}s", elapsed.as_secs_f64());
        let final_buf = &small_bufs[n_chain];
        let slice =
            unsafe { std::slice::from_raw_parts(final_buf.data() as *const f32, small_size) };
        let nz = slice.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
        let sum: f64 = slice.iter().map(|&v| v as f64).sum();
        let max_abs = slice.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        println!(
            "  mode={m:9} nz={nz}/{small_size} ({:.1}%) sum={sum:.4e} max_abs={max_abs:.4e}",
            100.0 * nz as f32 / small_size as f32
        );
        results.push((m, nz, sum));
    }

    // Compare: any mode with nz=0 while others non-zero = bug.
    let any_zero = results.iter().any(|&(_, nz, _)| nz == 0);
    let any_nonzero = results.iter().any(|&(_, nz, _)| nz > 0);
    if any_zero && any_nonzero {
        println!("--- BUG REPRODUCED: at least one mode gave all-zero output");
        println!("--- while another mode produced non-zero data");
        std::process::exit(2);
    } else if any_zero {
        println!("--- All modes gave zero (consistent failure or empty input?)");
    } else {
        println!("--- All modes produced non-zero output");
    }

    ctx.destroy_compute_pipeline(&mut conv_pipe);
    ctx.destroy_compute_pipeline(&mut slice_pipe);
    ctx.destroy_command_encoder(&mut encoder);
}
