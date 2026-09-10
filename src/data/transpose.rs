pub(super) fn transpose(data: &[f32], rows: usize, columns: usize, tile: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows.checked_mul(columns).unwrap());
    let mut output = vec![0.0; data.len()];
    if tile == 0 {
        for row in 0..rows {
            for column in 0..columns {
                output[column * rows + row] = data[row * columns + column];
            }
        }
    } else {
        for first_row in (0..rows).step_by(tile) {
            for first_column in (0..columns).step_by(tile) {
                for row in first_row..rows.min(first_row.saturating_add(tile)) {
                    for column in first_column..columns.min(first_column.saturating_add(tile)) {
                        output[column * rows + row] = data[row * columns + column];
                    }
                }
            }
        }
    }
    output
}
