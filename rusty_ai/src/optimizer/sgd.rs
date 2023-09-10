use super::OptimizerValues;
use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// configuration values for the stochastic gradient descent optimizer.
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

    fn init_with_layers(self, layers: &Vec<Layer>) -> Self::Optimizer {
        let prev_change = layers
            .iter()
            .map(Layer::init_zero_gradient)
            .collect::<Vec<_>>()
            .into();
        SGD_ { val: self, prev_change }
    }
}

/// stochastic gradient descent optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`SGD`] to create this optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SGD_ {
    val: SGD,
    prev_change: Gradient,
}

impl Optimizer for SGD_ {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        let SGD { learning_rate, momentum } = self.val;
        for ((x, dx), change) in nn
            .iter_mut()
            .zip(gradient.iter())
            .zip(self.prev_change.iter_mut())
        {
            *change = momentum * *change - learning_rate * dx;
            *x += *change;
        }
    }
}
