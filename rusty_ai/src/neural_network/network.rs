use crate::{matrix::Matrix, prelude::LayerBias};

pub type Gradient = Network<()>;

/// contains only weights and bias
#[derive(Debug, Clone)]
pub struct Layer<T> {
    weights: Matrix<f64>,
    bias: LayerBias,
    t: T,
}

/// contains only weights and biases
/// is also used for the gradient
#[derive(Debug, Clone)]
pub struct Network<T>(Vec<Layer<T>>);
