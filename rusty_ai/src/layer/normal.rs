use super::IsLayer;
use crate::{activation_function::ActivationFunction, matrix::Matrix};

#[derive(Debug)]
pub struct NormalLayer {
    weights: Matrix<f64>,
    bias: f64,
    activation_function: ActivationFunction,
}

impl NormalLayer {
    pub fn new(
        neuron_count: usize,
        next_neuron_count: usize,
        activation_function: ActivationFunction,
    ) -> NormalLayer {
        NormalLayer {
            weights: Matrix::new_random(neuron_count, next_neuron_count),
            bias: rand::random(),
            activation_function,
        }
    }
}

impl IsLayer for NormalLayer {
    fn get_neuron_count(&self) -> usize {
        *self.weights.get_width()
    }

    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .into_iter()
            .map(|x| x + self.bias)
            .map(&self.activation_function)
            .collect()
    }
}

impl std::fmt::Display for NormalLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Bias: {}; {}",
            self.weights, self.bias, self.activation_function
        )
    }
}
