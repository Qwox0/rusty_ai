use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::{const_matrix::Matrix, util::dot_product2};

trait MatrixBenchmarks<const W: usize, const H: usize> {
    fn to_vec(&self) -> Vec<usize>;
    fn collect(&self) -> [usize; W];
    fn for_for(&self) -> [usize; W];
    fn fold_for(&self) -> [usize; W];
}

impl<const W: usize, const H: usize> MatrixBenchmarks<W, H> for Matrix<f64, W, H> {
    fn to_vec(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; W], |acc: Vec<usize>, row| {
                acc.into_iter()
                    .zip(row)
                    .map(|(max, e)| std::cmp::max(max, e.to_string().len()))
                    .collect()
            })
    }

    fn collect(&self) -> [usize; W] {
        self.to_vec().try_into().unwrap()
    }

    fn for_for(&self) -> [usize; W] {
        let mut res = [0; W];
        for row in self.get_elements() {
            for (i, elem) in row.iter().enumerate() {
                res[i] = res[i].max(elem.to_string().len())
            }
        }
        res
    }

    fn fold_for(&self) -> [usize; W] {
        self.get_elements().iter().fold([0; W], |mut acc, row| {
            for (i, elem) in row.iter().enumerate() {
                acc[i] = acc[i].max(elem.to_string().len())
            }
            acc
        })
    }
}

make_benches! {
    Matrix<f64, MAX_COL_WIDTH_W, MAX_COL_WIDTH_H>;
    Matrix::new_random();
    to_vec
    collect
    for_for
    fold_for
}
