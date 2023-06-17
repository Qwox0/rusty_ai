use super::{IsOptimizer, DEFAULT_LEARNING_RATE};
use crate::{gradient::Gradient, neural_network::NeuralNetwork, traits::IterLayerParams};

// stochastic gradient descent
#[derive(Debug)]
pub struct GradientDescent {
    pub learning_rate: f64,
}

impl IsOptimizer for GradientDescent {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        for (x, dx) in nn.iter_mut_parameters().zip(gradient.iter_parameters()) {
            *x -= self.learning_rate * dx;
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
