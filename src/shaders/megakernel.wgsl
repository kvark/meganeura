// A workgroup owns a job's entire instruction stream and private arena.
// The queue only allocates immutable jobs; it never publishes tensor writes
// to another workgroup. No cross-workgroup barrier or residency assumption.
struct Job { base: u32, first: u32, count: u32, padding: u32 }
struct Instruction {
    op: u32, src: u32, other: u32, dst: u32,
    len: u32, columns: u32, scale: f32, bias: f32,
}
struct Params { jobs: u32, step: u32, padding0: u32, padding1: u32 }

var<storage> jobs: array<Job>;
var<storage> instructions: array<Instruction>;
var<storage> weights: array<f32>;
var<storage, read_write> arena: array<f32>;
// Head followed by one completion counter per job, for verification after wait.
var<storage, read_write> queue: array<atomic<u32>>;
var<storage, read_write> workers: array<vec2<u32>>;
var<uniform> params: Params;

var<workgroup> ticket: u32;
// Broadcast scalars/vectors: sharing a storage-layout struct type with
// workgroup memory triggers explicit-layout decorations in the SPIR-V backend.
var<workgroup> selected_job: vec4<u32>;
var<workgroup> instruction_header: vec4<u32>;
var<workgroup> instruction_body: vec4<u32>;
var<workgroup> sums: array<f32, 64>;

fn load_job(index: u32, lane: u32) -> Job {
    if lane == 0u {
        let job = jobs[index];
        selected_job = vec4<u32>(job.base, job.first, job.count, 0u);
    }
    let words = workgroupUniformLoad(&selected_job);
    return Job(words.x, words.y, words.z, words.w);
}

fn load_instruction(index: u32, lane: u32) -> Instruction {
    if lane == 0u {
        let inst = instructions[index];
        instruction_header = vec4<u32>(inst.op, inst.src, inst.other, inst.dst);
        instruction_body = vec4<u32>(inst.len, inst.columns,
            bitcast<u32>(inst.scale), bitcast<u32>(inst.bias));
    }
    let header = workgroupUniformLoad(&instruction_header);
    let body = workgroupUniformLoad(&instruction_body);
    return Instruction(header.x, header.y, header.z, header.w,
        body.x, body.y, bitcast<f32>(body.z), bitcast<f32>(body.w));
}

fn execute(inst: Instruction, base: u32, lane: u32) {
    switch inst.op {
        case 0u: { // affine
            for (var i = lane; i < inst.len; i += 64u) {
                arena[base + inst.dst + i] =
                    arena[base + inst.src + i] * inst.scale + inst.bias;
            }
        }
        case 1u: { // add
            for (var i = lane; i < inst.len; i += 64u) {
                arena[base + inst.dst + i] =
                    arena[base + inst.src + i] + arena[base + inst.other + i];
            }
        }
        case 2u: { // relu
            for (var i = lane; i < inst.len; i += 64u) {
                arena[base + inst.dst + i] = max(arena[base + inst.src + i], 0.0);
            }
        }
        case 3u: { // unweighted RMS normalization
            var sum = 0.0;
            for (var i = lane; i < inst.len; i += 64u) {
                let x = arena[base + inst.src + i];
                sum += x * x;
            }
            sums[lane] = sum;
            workgroupBarrier();
            for (var stride = 32u; stride > 0u; stride >>= 1u) {
                if lane < stride { sums[lane] += sums[lane + stride]; }
                workgroupBarrier();
            }
            let factor = inverseSqrt(sums[0] / f32(inst.len) + inst.bias);
            storageBarrier(); // also allow exact in-place normalization
            for (var i = lane; i < inst.len; i += 64u) {
                arena[base + inst.dst + i] = arena[base + inst.src + i] * factor;
            }
        }
        case 4u: { // row-major matrix times vector, shared read-only weights
            for (var row = lane; row < inst.len; row += 64u) {
                var sum = 0.0;
                for (var k = 0u; k < inst.columns; k++) {
                    sum += weights[inst.other + row * inst.columns + k]
                        * arena[base + inst.src + k];
                }
                arena[base + inst.dst + row] = sum;
            }
        }
        default: {}
    }
    storageBarrier();
}

@compute @workgroup_size(64)
fn persistent(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var completed = vec2<u32>(0u);
    loop {
        if lane == 0u { ticket = atomicAdd(&queue[0], 1u); }
        let index = workgroupUniformLoad(&ticket);
        if index >= params.jobs { break; }
        let job = load_job(index, lane);
        for (var pc = 0u; pc < job.count; pc++) {
            let inst = load_instruction(job.first + pc, lane);
            execute(inst, job.base, lane);
        }
        if lane == 0u { atomicAdd(&queue[index + 1u], 1u); }
        completed += vec2<u32>(1u, job.count);
    }
    if lane == 0u { workers[group.x] = completed; }
}

// Control: identical task bodies, one hardware dispatch per instruction depth.
@compute @workgroup_size(64)
fn separate(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let job = load_job(group.x, lane);
    if params.step < job.count {
        let inst = load_instruction(job.first + params.step, lane);
        execute(inst, job.base, lane);
    }
    if lane == 0u && params.step == max(job.count, 1u) - 1u {
        atomicAdd(&queue[group.x + 1u], 1u);
    }
}
