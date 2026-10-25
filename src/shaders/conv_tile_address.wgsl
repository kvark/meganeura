struct TileAddress { row: u32, column: u32, offset: u32 }

fn advance_address(value: TileAddress, grid: vec2<u32>, step: vec2<u32>, delta: vec3<u32>) -> TileAddress {
    var next = TileAddress(value.row + step.x, value.column + step.y, value.offset + delta.x);
    if next.column >= grid.y {
        next.column -= grid.y;
        next.row += 1u;
        next.offset += delta.y;
    }
    if next.row >= grid.x {
        next.row -= grid.x;
        next.offset += delta.z;
    }
    return next;
}

fn valid_address(value: TileAddress, lower: vec2<u32>, upper: vec2<u32>) -> bool {
    return all(vec4<bool>((value.row >= lower.x), (value.row < upper.x), (value.column >= lower.y), (value.column < upper.y)));
}
