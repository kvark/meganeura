// CPU arithmetic diagnostic; no GPU work or performance measurement.
// Shapes, structured inputs and gates from
// evidence/compensated-dw-accuracy-2026-09-06:tests/conv_derivatives.rs.
// Examine rounding of f64-evaluated dot-product partitions. This is not
// a simulation of a GPU shader, its accumulation order or its compiler.

fn main() {
    let scale = 1e-12_f32;
    for shape in [
        [1, 3, 224, 224, 64, 7, 7, 2, 3, 3],
        [1, 256, 56, 56, 64, 1, 1, 1, 0, 0],
        [3, 5, 7, 9, 7, 2, 3, 1, 1, 0],
        [2, 3, 1, 32771, 5, 1, 1, 1, 0, 0],
    ] {
        let [batch, ci, h, w, co, kh, kw, stride, ph, pw]: [usize; 10] = shape;
        let oh = (h + 2 * ph - kh) / stride + 1;
        let ow = (w + 2 * pw - kw) / stride + 1;
        let k = batch * oh * ow;
        let tiles = k.div_ceil(16);
        let nw = co * ci * kh * kw;
        let x: Vec<f32> = (0..batch * ci * h * w)
            .map(|i| 0.25 + (i % 29) as f32 / 1024.0)
            .collect();
        let dy: Vec<f32> = (0..batch * co * oh * ow)
            .map(|i| {
                let magnitude = 0.25 + (i % 29) as f32 / 1024.0;
                let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
                sign * magnitude * scale
            })
            .collect();
        // Cache f64 sums of each 16-product tile, one output at a time.
        let mut reference = vec![0.0_f64; nw];
        let counts = [1, 2, 3, 4, 8];
        let mut rounded_partitions = vec![vec![0.0_f32; nw]; counts.len()];
        let mut rounded_tiles = vec![0.0_f32; nw];
        let mut fma_tiles = vec![0.0_f32; nw];
        let mut unfused_tiles = vec![0.0_f32; nw];
        let mut tile_sums = vec![0.0_f64; tiles];
        let mut tile_fma = vec![0.0_f32; tiles];
        let mut tile_unfused = vec![0.0_f32; tiles];
        for output in 0..nw {
            tile_sums.fill(0.0);
            tile_fma.fill(0.0);
            tile_unfused.fill(0.0);
            let channel_out = output / (ci * kh * kw);
            let channel_in = output / (kh * kw) % ci;
            let kernel_y = output / kw % kh;
            let kernel_x = output % kw;
            for index in 0..k {
                let n = index / (oh * ow);
                let y = index / ow % oh;
                let z = index % ow;
                let iy = (y * stride + kernel_y) as isize - ph as isize;
                let ix = (z * stride + kernel_x) as isize - pw as isize;
                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                    continue;
                }
                let xi = ((n * ci + channel_in) * h + iy as usize) * w + ix as usize;
                let yi = ((n * co + channel_out) * oh + y) * ow + z;
                let product = f64::from(x[xi]) * f64::from(dy[yi]);
                reference[output] += product;
                tile_sums[index / 16] += product;
                tile_fma[index / 16] = x[xi].mul_add(dy[yi], tile_fma[index / 16]);
                tile_unfused[index / 16] += x[xi] * dy[yi];
            }
            rounded_tiles[output] =
                tile_sums.iter().map(|&v| f64::from(v as f32)).sum::<f64>() as f32;
            fma_tiles[output] = tile_fma.iter().copied().map(f64::from).sum::<f64>() as f32;
            unfused_tiles[output] = tile_unfused.iter().copied().map(f64::from).sum::<f64>() as f32;
            for (position, &splits) in counts.iter().enumerate() {
                let mut sum = 0.0;
                for split in 0..splits {
                    let first = split * (tiles / splits) + split.min(tiles % splits);
                    let last = first + tiles / splits + usize::from(split < tiles % splits);
                    let partial = tile_sums[first..last].iter().sum::<f64>() as f32;
                    sum += f64::from(partial);
                }
                rounded_partitions[position][output] = sum as f32;
            }
        }
        let norm: f64 = reference.iter().map(|v| v * v).sum();
        assert!(norm > 0.0);
        let report = |label: &str, actual: &[f32]| {
            let mut squared_error = 0.0;
            let mut rejected = 0;
            for (&a, &b) in actual.iter().zip(&reference) {
                let error = (f64::from(a) - b).abs();
                squared_error += error * error;
                rejected +=
                    usize::from(!a.is_finite() || error > f64::from(scale) * 1e-5 + b.abs() * 2e-4);
            }
            let relative_l2 = (squared_error / norm).sqrt();
            println!(
                "{shape:?} {label}: relative_l2={relative_l2:e}, rejected_elements={rejected}, pass={}",
                rejected == 0 && relative_l2 <= 2e-4
            );
        };
        report("rounded 16-product tiles + f64 sum", &rounded_tiles);
        report("f32 FMA 16-product tiles + f64 sum", &fma_tiles);
        report("f32 mul/add 16-product tiles + f64 sum", &unfused_tiles);
        for (splits, actual) in counts.into_iter().zip(rounded_partitions) {
            report(&format!("rounded {splits} partitions + f64 sum"), &actual);
        }
    }
}
