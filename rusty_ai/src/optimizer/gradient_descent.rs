use super::{IsOptimizer, DEFAULT_LEARNING_RATE};
use crate::{
    gradient::Gradient,
    neural_network::NeuralNetwork,
    util::{EntrySub, ScalarMul},
};

// stochastic gradient descent
#[derive(Debug)]
pub struct GradientDescent {
    pub learning_rate: f64,
}

impl IsOptimizer for GradientDescent {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: Gradient,
    ) {
        for (layer, mut gradient) in nn.iter_mut_layers().zip(gradient) {
            gradient.mul_scalar_mut(self.learning_rate);
            layer.sub_entries_mut(&gradient);
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
