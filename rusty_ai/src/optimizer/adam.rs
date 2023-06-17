use super::{IsOptimizer, DEFAULT_LEARNING_RATE};
use crate::gradient::Gradient;
use crate::traits::IterLayerParams;
use crate::util::{EntryDiv, EntrySub, Lerp, ScalarAdd};
use crate::{
    gradient::layer::GradientLayer,
    layer::Layer,
    matrix::Matrix,
    neural_network::NeuralNetwork,
    util::{constructor, EntryMul, ScalarMul},
};

// stochastic gradient descent
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<GradientLayer>,
    v: Vec<GradientLayer>,
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
        let time_step = nn.get_generation() + 1; // generation starts at 0. should start at 1
        todo!();
        /*
        for (((layer, lgradient), m), v) in nn
            .iter_mut_layers()
            .zip(gradient.iter_layers())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            m.lerp_mut(lgradient, self.beta1);
            v.lerp_mut(lgradient.sqare_entries(), self.beta2);

            let get_correction_factor = |beta: f64| (1.0 - beta.powi(time_step as i32)).recip();

            let m_bias_cor = m.clone().mul_scalar(get_correction_factor(self.beta1));
            let v_bias_cor = v.clone().mul_scalar(get_correction_factor(self.beta2));

            let denominator = v_bias_cor.sqrt_entries().add_scalar(self.epsilon);
            let change = m_bias_cor
                .mul_scalar(self.learning_rate)
                .mul_scalar(0.00001)
                .div_entries(denominator);

            //println!("{:?}", change);

            layer.sub_entries_mut(&change);
        }
        */
    }

    fn init_with_layers(&mut self, layers: &Vec<Layer>) {
        self.m = layers
            .iter()
            .map(|layer| {
                let (w, h) = layer.get_weights().get_dimensions();
                let weight_gradient = Matrix::with_zeros(w, h);
                let bias_gradient = layer.get_bias().clone_with_zeros();
                GradientLayer::new(weight_gradient, bias_gradient)
            })
            .collect();
        self.v = self.m.clone();
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![],
            v: vec![],
        }
    }
}
