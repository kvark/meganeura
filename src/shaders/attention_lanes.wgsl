// @section lane_sum
// Pairwise sum inside one query. The lane group is aligned in the subgroup
// because the lane count is a power of two no wider than the subgroup, so
// this is the tree reduction without its barriers.
fn lane_sum(value: f32) -> f32 {
    var sum = value;
$LANE_STEPS
    return sum;
}

// @section lane_step
    sum += subgroupShuffleXor(sum, $STRIDEu);

// @section partial_sums
var<workgroup> wg_part_score: array<f32, $THREADS>;
var<workgroup> wg_part_dp: array<f32, $THREADS>;
