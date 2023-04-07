use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::matrix::Matrix;

trait MatrixBenchmarks {
    fn mul(&self, rhs: Vec<f64>) -> Vec<f64>;
}

impl MatrixBenchmarks for Matrix<f64> {
    fn mul(&self, rhs: Vec<f64>) -> Vec<f64> {
        self * rhs
    }
}

make_benches! {
    Matrix<f64>;
    Matrix::new_random(MUL_SIZE_W, MUL_SIZE_H);
    mul: Vec::from(MUL_VEC)
}
