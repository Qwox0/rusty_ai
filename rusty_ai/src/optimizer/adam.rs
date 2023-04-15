use super::{Optimizer, DEFAULT_LEARNING_RATE};
use crate::{
    layer::Layer,
    matrix::Matrix,
    neural_network::{NNOptimizationParts, NeuralNetwork},
    results::GradientLayer,
    util::{macros::impl_new, EntryAdd, ScalarMul},
};

// stochastic gradient descent
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<Matrix<f64>>,
    m_bias: Vec<f64>,
    v: Vec<Matrix<f64>>,
    v_bias: Vec<f64>,
}

impl Adam {
    impl_new! {pub learning_rate: f64, beta1: f64, beta2: f64, epsilon:f64; Default}

    pub(crate) fn init_values(&mut self, layers: &Vec<Layer>) {
        let layer_count = layers.iter().skip(1).count();
        self.m = layers
            .iter()
            .skip(1) // Input layer isn't important
            .rev()
            .map(|l| l.get_weights().get_dimensions())
            .map(|(w, h)| Matrix::with_zeros(w, h))
            .collect();
        self.v = self.m.clone();
        self.m_bias = vec![0.0; layer_count];
        self.v_bias = vec![0.0; layer_count];
    }
}

impl Optimizer for Adam {
    fn optimize_weights<'a>(&mut self, nn: NNOptimizationParts, gradient: Vec<GradientLayer>) {
        let time_step = nn.generation + 1; // generation starts at 0. should start at 1
        for (((((layer, lgradient), m), m_bias), v), v_bias) in nn
            .layers
            .iter_mut()
            .skip(1)
            .rev()
            .zip(gradient)
            .zip(self.m.iter_mut())
            .zip(self.m_bias.iter_mut())
            .zip(self.v.iter_mut())
            .zip(self.v_bias.iter_mut())
        {
            // update m
            *m_bias *= self.beta1;
            *m_bias += lgradient.bias_change * (1.0 - self.beta1);
            m.mut_mul_scalar(self.beta1).mut_add_entries(
                lgradient
                    .weights_change
                    .clone()
                    .mul_scalar(1.0 - self.beta1),
            );

            // update v
            *v_bias *= self.beta2;
            *v_bias += lgradient.bias_change * lgradient.bias_change * (1.0 - self.beta2);
            let mut v_tmp = lgradient.weights_change.clone();
            for x in v_tmp.iter_mut() {
                *x *= *x;
            }
            v_tmp.mut_mul_scalar(1.0 - self.beta2);
            v.mut_mul_scalar(self.beta2).mut_add_entries(v_tmp);

            let v_bias_cor = *v_bias / (1.0 - self.beta2.powi(time_step as i32)) as f64;
            *layer.get_bias_mut() -= self.learning_rate / (self.epsilon + v_bias_cor.sqrt())
                * *m_bias
                / (1.0 - self.beta1.powi(time_step as i32)) as f64;
            for ((weight, m), v) in layer
                .get_weights_mut()
                .iter_mut()
                .zip(m.iter())
                .zip(v.iter())
            {
                // bias correction
                let m_bias_cor = m / (1.0 - self.beta1.powi(time_step as i32)) as f64;
                let v_bias_cor = v / (1.0 - self.beta2.powi(time_step as i32)) as f64;

                *weight -= self.learning_rate / (self.epsilon + v_bias_cor.sqrt()) * m_bias_cor;
            }
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.00000001,
            m: vec![],
            m_bias: vec![],
            v: vec![],
            v_bias: vec![],
        }
    }
}
