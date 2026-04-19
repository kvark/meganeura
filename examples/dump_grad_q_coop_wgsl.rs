//! Print the generated WGSL for the coop FlashGradQ kernel — useful
//! for debugging naga / SPIR-V translation issues.
fn main() {
    let sm = meganeura::codegen::generate_flash_grad_q_coop_module(64);
    println!("{}", sm.source);
}
