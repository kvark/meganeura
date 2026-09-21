use meganeura::codegen::ShaderModule;
use std::fmt::Write;

pub fn generate(
    block: [u32; 2],
    threads: [u32; 2],
    k: u32,
    interleave: bool,
    add: bool,
) -> ShaderModule {
    let [m, n] = block;
    let [x, y] = threads;
    let rows = m / y;
    let cols = n / x;
    let mut declarations = String::new();
    let mut compute = String::new();
    let mut stores = String::new();
    for i in 0..rows {
        writeln!(
            compute,
            "let a{i} = shared_a[(lid.y * {rows}u + {i}u) * {}u + kk];",
            k + 1
        )
        .unwrap();
    }
    for j in 0..cols {
        let col = if interleave {
            format!("lid.x + {j}u * {x}u")
        } else {
            format!("lid.x * {cols}u + {j}u")
        };
        writeln!(compute, "let b{j} = shared_b[kk * {}u + {col}];", n + 1).unwrap();
        for i in 0..rows {
            writeln!(declarations, "var s{i}_{j} = 0.0;").unwrap();
            writeln!(
                stores,
                "{{ let row = row0 + lid.y * {rows}u + {i}u; let col = col0 + {col};
                   if row < params.m && col < params.n {{ let index = row * params.n + col;
                   matrix_c[index] = s{i}_{j}{}; }} }}",
                if add { " + src[index]" } else { "" }
            )
            .unwrap();
        }
    }
    for i in 0..rows {
        for j in 0..cols {
            writeln!(compute, "s{i}_{j} = fma(a{i}, b{j}, s{i}_{j});").unwrap();
        }
    }
    let mut source = include_str!("scalar_layout.wgsl").to_owned();
    for (key, value) in [
        ("$M", m),
        ("$N", n),
        ("$K", k),
        ("$X", x),
        ("$Y", y),
        ("$THREADS", x * y),
        ("$A_STRIDE", k + 1),
        ("$B_STRIDE", n + 1),
        ("$A_SIZE", m * (k + 1)),
        ("$B_SIZE", k * (n + 1)),
    ] {
        source = source.replace(key, &format!("{value}u"));
    }
    source = source
        .replace("$DECLARATIONS", &declarations)
        .replace("$COMPUTE", &compute)
        .replace("$STORES", &stores)
        .replace(
            "$ADD",
            if add {
                "var<storage> src: array<f32>;"
            } else {
                ""
            },
        );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    ShaderModule {
        module,
        source,
        hint: "scalar_layout",
    }
}
