use crate::{
    activation_function::ActivationFunction::{self, *},
    matrix::Matrix,
    util::macros::{impl_getter, impl_new},
};

pub trait IsLayer: std::fmt::Display {
    fn get_neuron_count(&self) -> usize;
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64>;
}

#[derive(Debug)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
}
use LayerType::*;

/// Layer: all input weights + bias for all neurons in layer + activation function
/// The Propagation calculation is done in the same Order
#[derive(Debug)]
pub struct Layer {
    layer_type: LayerType,
    weights: Matrix<f64>,
    bias: f64,
    activation_function: ActivationFunction,
}

impl_getter! { Layer:
    get_layer_type -> layer_type: LayerType,
    get_weights -> weights: Matrix<f64>,
    get_bias -> bias: f64,
    get_activation_function -> activation_function: ActivationFunction,
}

impl_new! { Layer: layer_type: LayerType, weights: Matrix<f64>, bias: f64, activation_function: ActivationFunction }

impl Layer {
    pub fn new_input(neurons: usize) -> Layer {
        Layer::new(Input, Matrix::identity(neurons), 0.0, Identity)
    }

    pub fn new_hidden(inputs: usize, neurons: usize, acti_func: ActivationFunction) -> Layer {
        Layer::new(
            Hidden,
            Matrix::new_random(inputs, neurons),
            rand::random(),
            acti_func,
        )
    }

    pub fn new_output(inputs: usize, neurons: usize, acti_func: ActivationFunction) -> Layer {
        Layer::new(
            Output,
            Matrix::new_random(inputs, neurons),
            rand::random(),
            acti_func,
        )
    }

    pub fn get_input_count(&self) -> usize {
        *self.weights.get_width()
    }

    pub fn get_neuron_count(&self) -> usize {
        *self.weights.get_height()
    }

    /// An Input layer doesn't change the input, but still multiplies by the identity matrix and
    /// uses the identity activation function. It might be a good idea to skip the Input layer to
    /// reduce calculations.
    pub fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .into_iter()
            .map(|x| x + self.bias)
            .map(self.activation_function)
            .collect()
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plural_s = |x: usize| if x == 1 { "" } else { "s" };
        if let Input = self.layer_type {
            let input_count = self.get_input_count();
            write!(f, "{} Input{}\n", input_count, plural_s(input_count))?;
        }
        write!(
            f,
            "{} Bias: {}; {}",
            self.weights, self.bias, self.activation_function
        )?;
        if let Output = self.layer_type {
            let output_count = self.get_neuron_count();
            write!(f, "\n{} Output{}", output_count, plural_s(output_count))?;
        }
        Ok(())
    }
}
