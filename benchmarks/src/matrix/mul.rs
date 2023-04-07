use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::matrix::Matrix;
use rusty_ai::util::dot_product;

trait MatrixBenchmarks {
    fn mul(&self, rhs: Vec<f64>) -> Vec<f64>;
    fn mul_for(&self, rhs: Vec<f64>) -> Vec<f64>;
    fn mul_for_itermut(&self, rhs: Vec<f64>) -> Vec<f64>;
    fn mul_for_push(&self, rhs: Vec<f64>) -> Vec<f64>;
}

impl MatrixBenchmarks for Matrix<f64> {
    fn mul(&self, rhs: Vec<f64>) -> Vec<f64> {
        self.get_elements()
            .iter()
            .map(|row| dot_product(&row, &rhs))
            .collect::<Vec<f64>>()
    }

    fn mul_for(&self, rhs: Vec<f64>) -> Vec<f64> {
        let mut res = vec![0.0; *self.get_height()];
        for (i, row) in self.get_elements().iter().enumerate() {
            res[i] = dot_product(row, &rhs);
        }
        res
    }

    fn mul_for_itermut(&self, rhs: Vec<f64>) -> Vec<f64> {
        let mut res = vec![0.0; *self.get_height()];
        for (row, res) in self.get_elements().iter().zip(res.iter_mut()) {
            *res = dot_product(row, &rhs);
        }
        res
    }

    fn mul_for_push(&self, rhs: Vec<f64>) -> Vec<f64> {
        let mut res = Vec::with_capacity(*self.get_height());
        for row in self.get_elements() {
            res.push(dot_product(row, &rhs));
        }
        res
    }
}

make_benches! {
    Matrix<f64>;
    Matrix::new_random(MUL_SIZE_W, MUL_SIZE_H);
    mul(Vec::from(MUL_VEC))
    mul_for(Vec::from(MUL_VEC))
    mul_for_push(Vec::from(MUL_VEC))
}
