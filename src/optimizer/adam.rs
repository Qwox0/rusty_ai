//! Module containing the [`Adam_`] [`Optimizer`].

use super::DEFAULT_LEARNING_RATE;
use crate::{layer::Layer, util::Lerp, *};
use serde::{Deserialize, Serialize};

/// configuration values for the adam optimizer [`Adam_`].
///
/// use [`OptimizerValues::init_with_layers`] to create the optimizer: [`Adam_`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adam {
    /// The learning rate used by the adam optimizer.
    pub learning_rate: f64,
    /// The `beta1` constant used by the adam optimizer.
    pub beta1: f32,
    /// The `beta2` constant used by the adam optimizer.
    pub beta2: f32,
    /// The `epsilon` constant used by the adam optimizer.
    pub epsilon: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Self { learning_rate: DEFAULT_LEARNING_RATE, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
    }
}

impl<X: Float> OptimizerValues<X> for Adam {
    type Optimizer = Adam_<X>;

    fn init_with_layers(self, layers: &[Layer<X>]) -> Self::Optimizer {
        let v: Gradient<X> =
            layers.iter().map(Layer::init_zero_gradient).collect::<Vec<_>>().into();
        Adam_ { val: self, generation: 0, m: v.clone(), v }
    }
}

/// Adam Optimizer
///
/// this type implements [`Optimizer`]
///
/// use [`OptimizerValues::init_with_layers`] on [`Adam`] to create this optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adam_<X> {
    val: Adam,
    generation: usize,
    m: Gradient<X>,
    v: Gradient<X>,
}

impl<X: Float> Optimizer<X> for Adam_<X> {
    fn optimize<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<X, IN, OUT>,
        gradient: &Gradient<X>,
    ) {
        let Adam { learning_rate, beta1, beta2, epsilon } = self.val;
        self.generation += 1;
        let time_step = self.generation as i32;

        nn.iter_mut()
            .zip(gradient.iter())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
            .for_each(|(((x, dx), m), v)| {
                m.lerp_mut(*dx, beta1.cast());
                v.lerp_mut(*dx * *dx, beta2.cast());

                let m_bias_cor = *m / (1.0 - beta1.powi(time_step)).cast();
                let v_bias_cor = *v / (1.0 - beta2.powi(time_step)).cast();

                let denominator = v_bias_cor * v_bias_cor + epsilon.cast();
                *x -= m_bias_cor * learning_rate.cast() / denominator;
            });
    }
}
