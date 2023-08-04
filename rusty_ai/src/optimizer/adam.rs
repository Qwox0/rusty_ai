use crate::prelude::*;
use itertools::Itertools;

// stochastic gradient descent
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    generation: usize,
    m: Gradient,
    v: Gradient,
}

impl Adam {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        layers: &Vec<Layer>,
    ) -> Self {
        let m: Gradient = layers.iter().map(Layer::init_zero_gradient).collect_vec().into();
        let v = m.clone();

        Self { learning_rate, generation: 0, beta1, beta2, epsilon, m, v }
    }

    #[inline]
    pub fn with_learning_rate(learning_rate: f64, layers: &Vec<Layer>) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, layers)
    }

    #[inline]
    pub fn default(layers: &Vec<Layer>) -> Self {
        Self::with_learning_rate(DEFAULT_LEARNING_RATE, layers)
    }
}

impl IsOptimizer for Adam {
    fn optimize_weights<'a, const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    ) {
        self.generation += 1;
        let time_step = self.generation as i32;

        nn.iter_mut_parameters()
            .zip(gradient.iter_parameters())
            .zip(self.m.iter_mut_parameters())
            .zip(self.v.iter_mut_parameters())
            .for_each(|(((x, dx), m), v)| {
                m.lerp_mut(*dx, self.beta1);
                v.lerp_mut(dx * dx, self.beta2);

                let m_bias_cor = *m / (1.0 - self.beta1.powi(time_step));
                let v_bias_cor = *v / (1.0 - self.beta2.powi(time_step));

                let denominator = v_bias_cor * v_bias_cor + self.epsilon;
                *x -= m_bias_cor * self.learning_rate / denominator;
            });
    }
}
