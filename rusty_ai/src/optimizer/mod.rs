//! # Optimizer module

pub mod adam;
pub mod sgd;

use crate::{layer::Layer, *};

pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

pub trait Optimizer {
    fn optimize_weights<const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    );
}

pub trait OptimizerValues {
    type Optimizer: Optimizer;

    fn init_with_layers(self, layers: &[Layer]) -> Self::Optimizer;
}
