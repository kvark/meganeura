//! Run `runtime::auto_tune_flash_ept` against the connected GPU and
//! print the chosen EPT cap per flash kernel.
//!
//! Useful for verifying that the auto-tuner picks reasonable per-kernel
//! values on a given device. Doesn't install the result as a global.
//!
//! Usage:
//!   MEGANEURA_DEVICE_ID=<id> cargo run --release --example auto_tune_flash [head_dim]

fn main() {
    env_logger::init();
    let head_dim: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let dev_id = std::env::var("MEGANEURA_DEVICE_ID")
        .ok()
        .and_then(|s| s.parse().ok());
    let gpu = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            validation: false,
            timing: false,
            capture: false,
            overlay: false,
            device_id: dev_id,
            ..Default::default()
        })
    }
    .expect("failed to initialize GPU context");

    let info = gpu.device_information();
    eprintln!(
        "GPU: {:?} (driver={:?}, software={})",
        info.device_name, info.driver_name, info.is_software_emulated
    );
    eprintln!("Auto-tuning flash kernels at head_dim={head_dim}...\n");

    let cfg = meganeura::runtime::auto_tune_flash_ept(&gpu, head_dim);

    println!("FlashEptConfig (best EPT cap with regs ≤ 128):");
    println!("  forward:  {}", cfg.forward_cap);
    println!("  grad_q:   {}", cfg.grad_q_cap);
    println!("  grad_kv:  {}", cfg.grad_kv_cap);
    println!("  grad_k:   {}", cfg.grad_k_cap);
    println!("  grad_v:   {}", cfg.grad_v_cap);
}
