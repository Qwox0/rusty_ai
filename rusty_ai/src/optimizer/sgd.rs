//! Module containing the [`SGD_`] [`Optimizer`].
use super::{OptimizerValues, DEFAULT_LEARNING_RATE};
use crate::{layer::Layer, *};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// configuration values for the stochastic gradient descent optimizer [`SGD_`].
///
/// use [`OptimizerValues::init_with_layers`] to create the optimizer: [`SGD_`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
}

impl Default for SGD {
    fn default() -> Self {
        Self { learning_rate: DEFAULT_LEARNING_RATE, momentum: 0.0 }
    }
}

impl OptimizerValues for SGD {
    type Optimizer = SGD_;

    fn init_with_layers(self, layers: &[Layer]) -> Self::Optimizer {
        let prev_change = layers.iter().map(Layer::init_zero_gradient).collect::<Vec<_>>().into();
        SGD_ { val: self, prev_change }
    }
}

/// Stochastic gradient descent optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`SGD`] to create this
/// optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SGD_ {
    val: SGD,
    prev_change: Gradient,
}

impl Optimizer for SGD_ {
    fn optimize<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        let SGD { learning_rate, momentum } = self.val;
        for ((x, dx), change) in nn.iter_mut().zip(gradient.iter()).zip(self.prev_change.iter_mut())
        {
            *change = momentum * *change - learning_rate * dx;
            *x += *change;
        }
    }
}

impl Display for SGD_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let SGD { learning_rate, momentum } = self.val;
        write!(f, "SGD {{ learning_rate: {learning_rate}")?;
        if momentum != 0.0 {
            write!(f, ", momentum: {momentum}")?;
        }
        write!(f, " }}")
    }
}
