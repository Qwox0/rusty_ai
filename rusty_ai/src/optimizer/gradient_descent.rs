use super::{Optimizer, DEFAULT_LEARNING_RATE};
use crate::{
    layer::Layer,
    neural_network::{NNOptimizationParts, NeuralNetwork},
    results::GradientLayer,
    util::{EntrySub, ScalarMul},
};

// stochastic gradient descent
#[derive(Debug)]
pub struct GradientDescent {
    pub learning_rate: f64,
}

impl Optimizer for GradientDescent {
    fn optimize_weights<'a>(&mut self, nn: NNOptimizationParts, gradient: Vec<GradientLayer>) {
        for (layer, gradient) in nn.layers.iter_mut().skip(1).rev().zip(gradient) {
            *layer.get_bias_mut() -= self.learning_rate * gradient.bias_change;
            layer
                .get_weights_mut()
                .mut_sub_entries(gradient.weights_change.mul_scalar(self.learning_rate));
        }
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
        }
    }
}
