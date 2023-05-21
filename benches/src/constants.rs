pub fn fill_rand<T, const N: usize>(mut vec: [T; N]) -> Vec<T>
where
    T: Clone,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    vec.fill_with(rand::random);
    vec.to_vec()
}

pub const ITERATIONS: usize = 1;

pub const MATRIX_MUL_W: usize = 10000;
pub const MATRIX_MUL_H: usize = 10000;
pub const MATRIX_MUL_VEC: [f64; MATRIX_MUL_W] = [0.0; MATRIX_MUL_W];

pub const MAX_COL_WIDTH_W: usize = 20;
pub const MAX_COL_WIDTH_H: usize = 30;

pub const LAYER_CALC_IN: usize = 10000;
pub const LAYER_CALC_OUT: usize = 10000;
pub const LAYER_CALC_VEC: [f64; LAYER_CALC_IN] = [0.0; LAYER_CALC_IN];
