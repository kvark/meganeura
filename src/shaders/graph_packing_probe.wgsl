fn graph_packing_sync_probe() {
    let a = coop_mat16x16<f32, A, workgroup>();
    let b = coop_mat16x16<f32, B, workgroup>();
    let c = coop_mat16x16<f32, C, workgroup>();
    let product = coopMultiplyAdd(a, b, c);
}
