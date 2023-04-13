use crate::{
    matrix::Matrix,
    util::{macros::impl_new, EntryAdd, EntryDiv, ScalarMul},
};

/// Contains the estimated Gradient of the Costfunction with respect to the weights and the bias of
/// a layer in
#[derive(Debug)]
pub struct GradientLayer {
    pub bias_change: f64,
    pub weights_change: Matrix<f64>,
}

impl GradientLayer {
    impl_new! { pub weights_change: Matrix<f64>, bias_change: f64 }
    //impl_getter! { get_bias_change -> bias_change: f64 }
    //impl_getter! { get_weights_change -> weights_change: &Matrix<f64> }

    pub fn add_next_backpropagation(&mut self, dc_dweights: Matrix<f64>, sum_dc_dbias: f64) {
        self.bias_change += sum_dc_dbias;
        self.weights_change.mut_add_entries(&dc_dweights);
    }
}

impl ScalarMul for GradientLayer {
    fn mut_mul_scalar(&mut self, scalar: f64) -> &mut Self {
        self.bias_change *= scalar;
        self.weights_change.mut_mul_scalar(scalar);
        self
    }
}

