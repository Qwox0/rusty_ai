pub mod adam;
pub mod gradient_descent;

use self::{adam::Adam, gradient_descent::GradientDescent};
use crate::{layer::Layer, neural_network::{NeuralNetwork, NNOptimizationParts}, results::GradientLayer};

pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

pub trait Optimizer {
    fn optimize_weights<'a>(
        &mut self,
        nn: NNOptimizationParts,
        gradient: Vec<GradientLayer>,
    );
    fn init_with_layers(&mut self, layers: &Vec<Layer>) {}
}

#[derive(Debug)]
pub enum OptimizerDispatch {
    GradientDescent(GradientDescent),
    Adam(Adam),
}

impl OptimizerDispatch {
    pub fn gradient_descent(learning_rate: f64) -> OptimizerDispatch {
        OptimizerDispatch::GradientDescent(GradientDescent { learning_rate })
    }
    pub fn default_gradient_descent() -> OptimizerDispatch {
        OptimizerDispatch::GradientDescent(GradientDescent::default())
    }

    pub fn default_adam() -> OptimizerDispatch {
        OptimizerDispatch::Adam(Adam::default())
    }
}

impl Optimizer for OptimizerDispatch {
    fn optimize_weights<'a>(
        &mut self,
        nn: NNOptimizationParts,
        gradient: Vec<GradientLayer>,
    ) {
        match self {
            OptimizerDispatch::GradientDescent(gd) => gd.optimize_weights(nn, gradient),
            OptimizerDispatch::Adam(a) => a.optimize_weights(nn, gradient),
        }
    }

    fn init_with_layers(&mut self, layers: &Vec<Layer>) {
        match self {
            OptimizerDispatch::GradientDescent(gd) => gd.init_with_layers(layers),
            OptimizerDispatch::Adam(a) => a.init_with_layers(layers),
        }
    }
}
