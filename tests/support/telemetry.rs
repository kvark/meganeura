//! Shared checks for the continuous telemetry in the two instrumented cohorts.
use serde_json::Value;

pub fn time_window(value: &Value, earliest: u64, latest: u64) -> u64 {
    let host = value["host_start"]["unix_ms"].as_u64().unwrap();
    let start = value["started_unix_ms"].as_u64().unwrap();
    let end = value["finished_unix_ms"].as_u64().unwrap();
    assert!(earliest <= host && host <= start && start < end && end <= latest);
    end
}

pub fn check_telemetry(record: &Value) {
    let telemetry = &record["telemetry"];
    assert!(telemetry["error"].is_null());
    assert_eq!(telemetry["requested_interval_ms"], 250);
    assert_eq!(telemetry["sample_cap"], 40000);
    assert_eq!(telemetry["cap_reached"], false);
    assert_eq!(
        telemetry["fields"],
        "timestamp,uuid,utilization.gpu,memory.used,clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate"
    );
    let samples = telemetry["samples"].as_array().unwrap();
    assert!(!samples.is_empty());
    let mut previous = 0;
    for sample in samples {
        let time = sample["received_unix_ms"].as_u64().unwrap();
        assert!(time >= previous);
        previous = time;
        let fields: Vec<_> = sample["csv"]
            .as_str()
            .unwrap()
            .split(',')
            .map(str::trim)
            .collect();
        assert_eq!(fields.len(), 9);
        assert_eq!(fields[1], "GPU-705c613d-97a2-2380-4fd9-49006cebab54");
        for field in &fields[2..8] {
            let number = field.parse::<f64>().unwrap();
            assert!(number.is_finite() && number >= 0.0);
        }
        assert!(fields[2].parse::<f64>().unwrap() <= 100.0);
        assert!(fields[8].strip_prefix('P').unwrap().parse::<u32>().is_ok());
    }
    assert!(
        samples[0]["received_unix_ms"].as_u64().unwrap()
            <= record["metadata"]["started_unix_ms"].as_u64().unwrap()
    );
    assert!(previous <= record["finished_unix_ms"].as_u64().unwrap());
}
