// @section bindings
var<storage> $B_BUFFER: $B_STORAGE;
var<storage, read_write> $C_BUFFER: array<f32>;

// @section entry_signature
@compute @workgroup_size($WORKGROUP_SIZE)
fn $NAME(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$SUBGROUP)

// @section helper_signature
fn $NAME(wgid: vec3<u32>, lid: vec3<u32>$SUBGROUP)

// @section call
    $CONDITION { $NAME(wgid, lid$SUBGROUP_ARG); }

// @section dispatch
$ENTRY_SIGNATURE {
$CALLS
}
