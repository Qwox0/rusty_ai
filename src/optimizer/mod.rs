//! # Optimizer module

use crate::nn::NNComponent;
use const_tensor::{Element, Shape, Tensor};
//pub mod adam;
//pub mod sgd;

/// Default learning rate used by [`Optimizer`]s in `rusty_ai::optimizer::*`.
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Optimizes the parameters of a [`NeuralNetwork`] based on a [`Gradient`].
///
/// This is only used by [`NNTrainer`]. Thus the dimensions of `nn` and `gradient` will always
/// match.
pub trait Optimizer<X: Element> {
    type State;

    /// Optimize the parameters of a [`NeuralNetwork`] based on a [`Gradient`].
    fn optimize<IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT>>(
        &self,
        nn: C,
        gradient: C::Grad,
        state: Self::State,
    ) -> C;
}

/*
/// Represents the constants/configuration of an [`Optimizer`].
///
/// Used by [`NNTrainerBuilder`] to create an [`Optimizer`] of type `Self::Optimizer`.
pub trait OptimizerValues<X> {
    /// Target [`Optimizer`] type
    type Optimizer: Optimizer<X>;

    /// Creates an [`Optimizer`] based on the configuration in `self` and the [`NeuralNetwork`]
    /// [`Layer`]s in `layers`.
    fn init_with_layers(self, layers: &[Layer<X>]) -> Self::Optimizer;
}
*/
