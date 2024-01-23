//! Module containing the [`Adam_`] [`Optimizer`].

use super::{Optimizer, DEFAULT_LEARNING_RATE};
use const_tensor::{Element, Float, Len, Num, Shape, Tensor};
use serde::{Deserialize, Serialize};

/// Adam Optimizer
///
/// This type implements [`Optimizer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adam<X> {
    /// The learning rate used by the adam optimizer.
    pub learning_rate: X,
    /// The `beta1` constant used by the adam optimizer.
    pub beta1: X,
    /// The `beta2` constant used by the adam optimizer.
    pub beta2: X,
    /// The `epsilon` constant used by the adam optimizer.
    pub epsilon: X,
}

impl<X: Float> Default for Adam<X> {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE.cast(),
            beta1: 0.9.cast(),
            beta2: 0.999.cast(),
            epsilon: 1e-8.cast(),
        }
    }
}

pub struct AdamState<X: Element, S: Shape> {
    m: Tensor<X, S>,
    v: Tensor<X, S>,
    generation: usize,
}

impl<X: Float> Optimizer<X> for Adam<X> {
    type State<S: Shape> = AdamState<X, S>;

    fn optimize_tensor<S: Shape + Len<LEN>, const LEN: usize>(
        &self,
        tensor: &mut Tensor<X, S>,
        gradient: &const_tensor::tensor<X, S>,
        state: Self::State<S>,
    ) -> Self::State<S> {
        let Adam { learning_rate, beta1, beta2, epsilon } = *self;

        let generation = state.generation + 1;
        let m = state.m.lerp(gradient, beta1);
        let v = state.v.lerp(&gradient.to_owned().square_elem(), beta2);

        let m_bias_cor = m.clone().scalar_div(X::ONE - beta1.powi(generation as i32));
        let v_bias_cor = v.clone().scalar_div(X::ONE - beta2.powi(generation as i32));

        let denominator = v_bias_cor.square_elem().scalar_add(epsilon);
        let change = m_bias_cor.scalar_mul(learning_rate).div_elem(&denominator);
        tensor.sub_elem_mut(&change);
        AdamState { m, v, generation }
    }

    fn new_state<S: Shape>(tensor: Tensor<X, S>) -> Self::State<S> {
        let m = tensor.clone();
        AdamState { m, v: tensor, generation: 0 }
    }
}
