use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::{matrix::Matrix, util::dot_product};

trait MatrixBenchmarks {
    fn v1(&self) -> Vec<usize>;
    fn v2(&self) -> Vec<usize>;
    fn v3(&self) -> Vec<usize>;
    fn v4(&self) -> Vec<usize>;
    fn v5(&self) -> Vec<usize>;
}

impl MatrixBenchmarks for Matrix<f64> {
    fn v1(&self) -> Vec<usize> {
        let mut column_widths = vec![0; *self.get_width()];
        for row in self.get_elements().iter() {
            for (idx, element) in row.iter().enumerate() {
                column_widths[idx] = element.to_string().len().max(column_widths[idx]);
            }
        }
        column_widths
    }

    fn v2(&self) -> Vec<usize> {
        let mut column_widths = vec![0; *self.get_width()];
        for row in self.get_elements().iter() {
            for (max, element) in column_widths.iter_mut().zip(row) {
                *max = element.to_string().len().max(*max)
            }
        }
        column_widths
    }

    fn v3(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |mut acc, row| {
                for (max, element) in acc.iter_mut().zip(row) {
                    *max = element.to_string().len().max(*max)
                }
                acc
            })
    }

    fn v4(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |acc, row| {
                row.iter()
                    .zip(acc)
                    .map(|(e, max)| std::cmp::max(max, e.to_string().len()))
                    .collect()
            })
    }

    fn v5(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |acc, row| {
                acc.into_iter()
                    .zip(row)
                    .map(|(max, e)| std::cmp::max(max, e.to_string().len()))
                    .collect()
            })
    }
}

make_benches! {
    Matrix<f64>;
    Matrix::new_random(MAX_COL_WIDTH_W, MAX_COL_WIDTH_H);
    v1
    v2
    v3
    v4
    v5
}
