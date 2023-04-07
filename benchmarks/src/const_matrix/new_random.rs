use crate::constants::*;
use crate::macros::make_benches;
use rand::Rng;
use rusty_ai::const_matrix::Matrix;

trait MatrixBenchmarks<T> {
    fn arr_map(_: &i32);
    fn arr_for(_: &i32);
}

impl MatrixBenchmarks<f64> for Matrix<f64, NEW_RAND_W, NEW_RAND_H> {
    fn arr_map(_: &i32) {
        let mut rng = rand::thread_rng();
        let a: Self =
            Matrix::from_rows([[0; NEW_RAND_W]; NEW_RAND_H].map(|row| row.map(|_| rng.gen())));
    }

    fn arr_for(_: &i32) {
        let mut rng = rand::thread_rng();
        let a: Self = Matrix::from_rows({
            let mut a = [[0.0; NEW_RAND_W]; NEW_RAND_H];
            for row in a.iter_mut() {
                for elem in row.iter_mut() {
                    *elem = rng.gen();
                }
            }
            a
        });
    }
}

make_benches! {
    Matrix<f64, NEW_RAND_W, NEW_RAND_H>;
    0;
    arr_map
    arr_for
}
