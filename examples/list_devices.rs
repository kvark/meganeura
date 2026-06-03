//! Print every compute device meganeura can address from this process.
//!
//! Default build only enumerates the Blade GPU. Build with
//! `--features openvino` to additionally probe the OpenVINO runtime for
//! Intel NPUs (requires `libopenvino` to be installed on the host).

use meganeura::list_devices;

fn main() {
    env_logger::init();

    let devices = list_devices();
    if devices.is_empty() {
        println!("no compute devices found");
        return;
    }

    println!(
        "{:<11} {:<10} {:<6} {}",
        "kind", "id", "train", "name"
    );
    for d in &devices {
        println!(
            "{:<11} {:<10} {:<6} {}",
            d.kind.as_str(),
            d.backend_id,
            if d.supports_training { "yes" } else { "no" },
            d.name,
        );
    }
}
