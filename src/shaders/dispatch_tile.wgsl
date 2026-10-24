@compute @workgroup_size($WORKGROUP)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    execute_tile(wgid, lid);
}
