use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::matrix::Matrix;

trait MatrixBenchmarks<T> {
    fn bench_new_random(_: &i32);
}

impl MatrixBenchmarks<f64> for Matrix<f64> {
    fn bench_new_random(_: &i32) {
        let a = Matrix::new_random(NEW_RAND_W, NEW_RAND_H);
    }
}

make_benches! {
    Matrix<f64>;
    0;
    bench_new_random
}
