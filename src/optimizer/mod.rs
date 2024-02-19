//! # Optimizer module

use const_tensor::{Element, Shape, Tensor};
pub mod adam;
pub mod sgd;

/// Default learning rate used by [`Optimizer`]s in `rusty_ai::optimizer::*`.
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Optimizes the parameters of a [`NeuralNetwork`] based on a [`Gradient`].
///
/// This is only used by [`NNTrainer`]. Thus the dimensions of `nn` and `gradient` will always
/// match.
pub trait Optimizer<X: Element>: Send + Sync + 'static {
    /// State required for optimizing a [`Tensor`] with [`Shape`] `S`.
    type State<S: Shape>: Sized + Send + Sync + 'static;

    /// optimizes a [`Tensor`].
    fn optimize_tensor<S: Shape>(
        &self,
        tensor: &mut Tensor<X, S>,
        gradient: &const_tensor::tensor<X, S>,
        state: &mut Self::State<S>,
    );

    /// Creates a new optimizer state from a tensor.
    fn new_state<S: Shape>(tensor: Tensor<X, S>) -> Self::State<S>;
}
