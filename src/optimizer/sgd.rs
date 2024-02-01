//! Module containing the [`SGD_`] [`Optimizer`].

use super::{Optimizer, DEFAULT_LEARNING_RATE};
use const_tensor::{
    Element, Len, Multidimensional, MultidimensionalOwned, Num, Shape, Tensor, VectorShape,
};
use core::fmt;
use serde::{Deserialize, Serialize};

/// Stochastic gradient descent optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`SGD`] to create this
/// optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SGD<X> {
    /// The learning rate used by the sgd optimizer.
    pub learning_rate: X,
    /// The `momentum` constant used by the sgd optimizer.
    pub momentum: X,
}

impl<X: Num> Default for SGD<X> {
    fn default() -> Self {
        Self { learning_rate: DEFAULT_LEARNING_RATE.cast(), momentum: X::ZERO }
    }
}

pub struct SGDState<X: Element, S: Shape> {
    prev_grad: Tensor<X, S>,
}

impl<X: Num> Optimizer<X> for SGD<X> {
    type State<S: Shape> = SGDState<X, S>;

    fn optimize_tensor<S: Shape>(
        &self,
        tensor: &mut Tensor<X, S>,
        gradient: &const_tensor::tensor<X, S>,
        state: &mut Self::State<S>,
    ) {
        state.prev_grad.scalar_mul_mut(self.momentum);
        state
            .prev_grad
            .sub_elem_mut(&gradient.to_owned().scalar_mul(self.learning_rate));
        tensor.add_elem_mut(&state.prev_grad);
    }

    fn new_state<S: Shape>(tensor: Tensor<X, S>) -> Self::State<S> {
        SGDState { prev_grad: tensor }
    }
}

impl<X: Num> fmt::Display for SGD<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let SGD { learning_rate, momentum } = self;
        write!(f, "SGD {{ learning_rate: {:?}", learning_rate)?;
        if *momentum != X::ZERO {
            write!(f, ", momentum: {:?}", momentum)?;
        }
        write!(f, " }}")
    }
}

/*
use super::{Optimizer, OptimizerValues, DEFAULT_LEARNING_RATE};
use crate::{layer::Layer, *};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/*
/// configuration values for the stochastic gradient descent optimizer [`SGD_`].
///
/// use [`OptimizerValues::init_with_layers`] to create the optimizer: [`SGD_`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SGD {
    /// The learning rate used by the sgd optimizer.
    pub learning_rate: f64,
    /// The `momentum` constant used by the sgd optimizer.
    pub momentum: f64,
}

impl Default for SGD {
    fn default() -> Self {
        Self { learning_rate: DEFAULT_LEARNING_RATE, momentum: 0.0 }
    }
}

impl<X: Float> OptimizerValues<X> for SGD {
    type Optimizer = SGD_<X>;

    fn init_with_layers(self, layers: &[Layer<X>]) -> Self::Optimizer {
        let prev_change = layers.iter().map(Layer::init_zero_gradient).collect::<Vec<_>>().into();
        SGD_ { val: self, prev_change }
    }
}
*/

/// Stochastic gradient descent optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`SGD`] to create this
/// optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SGD_<X> {
    val: SGD,
    prev_change: Gradient<X>,
}

impl<X: Num> Optimizer<X> for SGD_<X> {
    type State;

    fn optimize<IN: const_tensor::Tensor<X>, OUT: const_tensor::Tensor<X>, C: nn::NNComponent<X, IN, OUT>>(
        &self,
        nn: C,
        gradient: C::Grad,
        state: Self::State,
    ) -> C {
        todo!()
    }

    /*
    fn optimize<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<X, IN, OUT>,
        gradient: &Gradient<X>,
    ) {
        let SGD { learning_rate, momentum } = self.val;
        for ((x, dx), change) in nn.iter_mut().zip(gradient.iter()).zip(self.prev_change.iter_mut())
        {
            *change = momentum.cast::<X>() * *change - learning_rate.cast::<X>() * *dx;
            *x += *change;
        }
    }
    */
}

impl<X> Display for SGD_<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let SGD { learning_rate, momentum } = self.val;
        write!(f, "SGD {{ learning_rate: {learning_rate}")?;
        if momentum != 0.0 {
            write!(f, ", momentum: {momentum}")?;
        }
        write!(f, " }}")
    }
}
*/
