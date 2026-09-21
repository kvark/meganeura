use std::{hint::black_box, time::Instant};

#[derive(Clone, Copy, Debug, Default)]
struct Error {
    max_error: f64,
    max_reference: f64,
    square_error: f64,
    square_reference: f64,
}

fn serial(actual: &[f32], reference: &[f32]) -> Result<Error, ()> {
    let mut out = Error::default();
    for (&a, &b) in actual.iter().zip(reference) {
        if !a.is_finite() || !b.is_finite() {
            return Err(());
        }
        let (a, b) = (f64::from(a), f64::from(b));
        out.max_error = out.max_error.max((a - b).abs());
        out.max_reference = out.max_reference.max(b.abs());
        out.square_error += (a - b).powi(2);
        out.square_reference += b * b;
    }
    Ok(out)
}

fn parallel_lanes(actual: &[f32], reference: &[f32]) -> Result<Error, ()> {
    let mut lanes = [Error::default(); 8];
    let mut ac = actual.chunks_exact(8);
    let mut rc = reference.chunks_exact(8);
    for (a, b) in ac.by_ref().zip(rc.by_ref()) {
        let mut finite = true;
        for i in 0..8 {
            finite &= a[i].is_finite() && b[i].is_finite();
            let (a, b) = (f64::from(a[i]), f64::from(b[i]));
            lanes[i].max_error = lanes[i].max_error.max((a - b).abs());
            lanes[i].max_reference = lanes[i].max_reference.max(b.abs());
            lanes[i].square_error += (a - b).powi(2);
            lanes[i].square_reference += b * b;
        }
        if !finite {
            return Err(());
        }
    }
    let mut out = serial(ac.remainder(), rc.remainder())?;
    for lane in lanes {
        out.max_error = out.max_error.max(lane.max_error);
        out.max_reference = out.max_reference.max(lane.max_reference);
        out.square_error += lane.square_error;
        out.square_reference += lane.square_reference;
    }
    Ok(out)
}

fn main() {
    for count in [127, 1_000_003, 25_000_003] {
        let reference: Vec<f32> = (0..count)
            .map(|i| ((i as f64 * 0.37).sin() * 0.0001) as f32)
            .collect();
        let mut actual: Vec<f32> = reference
            .iter()
            .enumerate()
            .map(|(i, &x)| x + (i % 7) as f32 * 1e-11)
            .collect();
        let baseline = serial(&actual, &reference).unwrap();
        let challenger = parallel_lanes(&actual, &reference).unwrap();
        assert_eq!(baseline.max_error, challenger.max_error);
        assert_eq!(baseline.max_reference, challenger.max_reference);
        assert!(
            (baseline.square_error - challenger.square_error).abs() < baseline.square_error * 1e-9
        );
        assert!(
            (baseline.square_reference - challenger.square_reference).abs()
                < baseline.square_reference * 1e-9
        );
        for sample in 0..7 {
            for (name, function) in [
                ("serial", serial as fn(&[f32], &[f32]) -> _),
                ("lanes", parallel_lanes),
            ] {
                let start = Instant::now();
                for _ in 0..4 {
                    black_box(function(black_box(&actual), black_box(&reference)).unwrap());
                }
                println!(
                    "{{\"count\":{count},\"sample\":{sample},\"mode\":\"{name}\",\"ms\":{}}}",
                    start.elapsed().as_secs_f64() * 250.0
                );
            }
        }
        for index in [0, 7, 8, count - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let previous = actual[index];
                actual[index] = value;
                assert!(parallel_lanes(&actual, &reference).is_err());
                actual[index] = previous;
            }
        }
    }
}
