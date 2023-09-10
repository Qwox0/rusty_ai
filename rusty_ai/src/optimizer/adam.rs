use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// configuration values for the stochastic gradient descent optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for Adam {
    fn default() -> Self {
        Self { learning_rate: DEFAULT_LEARNING_RATE, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
    }
}

impl OptimizerValues for Adam {
    type Optimizer = Adam_;

    fn init_with_layers(self, layers: &[Layer]) -> Self::Optimizer {
        let v: Gradient = layers
            .iter()
            .map(Layer::init_zero_gradient)
            .collect::<Vec<_>>()
            .into();
        Adam_ { val: self, generation: 0, m: v.clone(), v }
    }
}

/// Adam Optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`Adam`] to create this optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adam_ {
    val: Adam,
    generation: usize,
    m: Gradient,
    v: Gradient,
}

impl Optimizer for Adam_ {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        let Adam { learning_rate, beta1, beta2, epsilon } = self.val;
        self.generation += 1;
        let time_step = self.generation as i32;

        nn.iter_mut()
            .zip(gradient.iter())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
            .for_each(|(((x, dx), m), v)| {
                m.lerp_mut(*dx, beta1);
                v.lerp_mut(dx * dx, beta2);

                let m_bias_cor = *m / (1.0 - beta1.powi(time_step));
                let v_bias_cor = *v / (1.0 - beta2.powi(time_step));

                let denominator = v_bias_cor * v_bias_cor + epsilon;
                *x -= m_bias_cor * learning_rate / denominator;
            });
    }
}
