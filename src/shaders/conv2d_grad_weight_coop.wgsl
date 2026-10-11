// One kernel offset per workgroup: dW[Co,Ci,kh,kw] = dY * shifted(X)^T.
// Both operands stage contiguous spatial positions, then B transposes in
// shared memory. Split-K partials retain the ordinary SumRows layout.
$ENABLE_F16
enable wgpu_cooperative_matrix;
$PARAMS
$DIVISOR
$DIGITS

var<storage> grad_out: array<f32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> a: array<$ELEM, $OPERAND_SIZE>;
var<workgroup> b: array<$ELEM, $OPERAND_SIZE>;
var<workgroup> result: array<f32, $RESULT_SIZE>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) counts: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(subgroup_id) sg: u32) {
    let row = wg.x * $WIDTHu;
    let col = wg.y * $WIDTHu;
    let m = params.out_channels;
    let kernel_hw = params.kernel_h * params.kernel_w;
    let n = params.in_channels * kernel_hw;
    let split_id = wg.z / kernel_hw;
    let offset = wg.z % kernel_hw;
    let kh = offset / params.kernel_w;
    let kw = offset % params.kernel_w;
    let splits = counts.z / kernel_hw;
    let spatial = params.out_h * params.out_w;
    let k = params.batch * spatial;
    let tiles = (k + $TILEu - 1u) / $TILEu;
    let per_split = tiles / splits;
    let extra = tiles % splits;
    let first = split_id * per_split + min(split_id, extra);
    let end = min((first + per_split + u32(split_id < extra)) * $TILEu, k);
    let input_spatial = params.in_h * params.in_w;
    var acc00 = coop_mat$TILEx$TILE<f32,C>();
    var acc01 = coop_mat$TILEx$TILE<f32,C>();
    var acc10 = coop_mat$TILEx$TILE<f32,C>();
    var acc11 = coop_mat$TILEx$TILE<f32,C>();

    for (var t = first * $TILEu; t < end; t += $TILEu) {
        let ka = t + tid % $TILEu;
        let pos = split_digits(ka, params.out_h, params.out_w,
            params.column_width_multiplier, params.output_spatial_multiplier);
        let ih = i32(pos.middle * params.stride + kh) - i32(params.padding_h);
        let iw = i32(pos.inner * params.stride + kw) - i32(params.padding_w);
        for (var index = tid; index < $OPERAND_SIZEu; index += 64u) {
            let co = row + index / $TILEu;
            var av = 0.0;
            if co < m && ka < end {
                av = grad_out[(pos.outer * m + co) * spatial + pos.middle * params.out_w + pos.inner];
            }
            a[index] = $ELEM(av);

            let ci = col + index / $TILEu;
            var bv = 0.0;
            if ci < params.in_channels && ka < end && ih >= 0 && iw >= 0
                && u32(ih) < params.in_h && u32(iw) < params.in_w {
                bv = src[(pos.outer * params.in_channels + ci) * input_spatial
                    + u32(ih) * params.in_w + u32(iw)];
            }
            b[(index % $TILEu) * $WIDTHu + index / $TILEu] = $ELEM(bv);
        }
        workgroupBarrier();
        let a0 = coopLoadT<coop_mat$TILEx$TILE<$ELEM,A>>(&a[0], $TILEu);
        let a1 = coopLoadT<coop_mat$TILEx$TILE<$ELEM,A>>(&a[$TILEu * $TILEu], $TILEu);
        let b0 = coopLoadT<coop_mat$TILEx$TILE<$ELEM,B>>(&b[0], $WIDTHu);
        let b1 = coopLoadT<coop_mat$TILEx$TILE<$ELEM,B>>(&b[$TILEu], $WIDTHu);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);
        workgroupBarrier();
    }
    if sg == 0u {
        coopStoreT(acc00, &result[0], $WIDTHu);
        coopStoreT(acc01, &result[$TILEu], $WIDTHu);
        coopStoreT(acc10, &result[$TILEu * $WIDTHu], $WIDTHu);
        coopStoreT(acc11, &result[$TILEu * $WIDTHu + $TILEu], $WIDTHu);
    }
    workgroupBarrier();
    for (var index = tid; index < $RESULT_SIZEu; index += 64u) {
        let r = row + index / $WIDTHu;
        let c = col + index % $WIDTHu;
        if r < m && c < params.in_channels {
            dst[(split_id * m + r) * n + c * kernel_hw + offset] = result[index];
        }
    }
}
