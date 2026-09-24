// A mixed-radix index `(outer * middle_n + middle) * inner_n + inner`, such as
// (channel, kernel row, kernel column) or (image, output row, output column).
// Staging loops advance one by a constant step with carries instead of
// dividing every element's flat index.
struct Digits {
    outer: u32,
    middle: u32,
    inner: u32,
}

fn split_digits(value: u32, middle_n: u32, inner_n: u32, inner_multiplier: u32, block_multiplier: u32) -> Digits {
    let block = middle_n * inner_n;
    let outer = divide_exact(value, block, block_multiplier);
    let rest = value - outer * block;
    let middle = divide_exact(rest, inner_n, inner_multiplier);
    return Digits(outer, middle, rest - middle * inner_n);
}

// Both digits of `step` are below their radices, so one carry each suffices.
fn advance_digits(value: Digits, step: Digits, middle_n: u32, inner_n: u32) -> Digits {
    var inner = value.inner + step.inner;
    var middle = value.middle + step.middle;
    let inner_wraps = inner >= inner_n;
    inner = select(inner, inner - inner_n, inner_wraps);
    middle += u32(inner_wraps);
    let middle_wraps = middle >= middle_n;
    middle = select(middle, middle - middle_n, middle_wraps);
    return Digits(value.outer + step.outer + u32(middle_wraps), middle, inner);
}
