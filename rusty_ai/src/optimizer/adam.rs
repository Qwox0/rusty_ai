use itertools::Itertools;

use super::{IsOptimizer, DEFAULT_LEARNING_RATE};
use crate::gradient::Gradient;
use crate::traits::IterLayerParams;
use crate::util::Lerp;
use crate::{layer::Layer, neural_network::NeuralNetwork, util::constructor};

// Markers
#[derive(Debug)]
struct AdamData {
    m: Gradient,
    v: Gradient,
}

// stochastic gradient descent
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    data: Option<AdamData>,
}

impl Adam {
    constructor! { pub new -> learning_rate: f64, beta1: f64, beta2: f64, epsilon:f64; Default }
    constructor! { pub with_learning_rate -> learning_rate: f64; Default }
}

impl IsOptimizer for Adam {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        let time_step = (nn.get_generation() + 1) as i32; // generation starts at 0. should start at 1
        let AdamData { m, v } = self
            .data
            .as_mut()
            .expect("Adam Optimizer needs to be initialized with layers first!");

        nn.iter_mut_parameters()
            .zip(gradient.iter_parameters())
            .zip(m.iter_mut_parameters())
            .zip(v.iter_mut_parameters())
            .for_each(|(((x, dx), m), v)| {
                m.lerp_mut(*dx, self.beta1);
                v.lerp_mut(dx * dx, self.beta2);

                let m_bias_cor = *m / (1.0 - self.beta1.powi(time_step));
                let v_bias_cor = *v / (1.0 - self.beta2.powi(time_step));

                let denominator = v_bias_cor * v_bias_cor + self.epsilon;
                *x -= m_bias_cor * self.learning_rate / denominator;
            });
    }

    fn init_with_layers(&mut self, layers: &Vec<Layer>) {
        let m: Gradient = layers
            .iter()
            .map(Layer::init_zero_gradient)
            .collect_vec()
            .into();
        let v = m.clone();
        let _ = self.data.insert(AdamData { m, v });
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            data: None,
        }
    }
}
