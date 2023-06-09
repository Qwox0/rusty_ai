mod bias;
use crate::{
    activation_function::ActivationFn,
    gradient::aliases::{InputGradient, OutputGradient},
    gradient::layer::GradientLayer,
    matrix::Matrix,
    traits::{impl_IterParams, IterParams},
    util::{impl_getter, EntryAdd},
};
pub use bias::*;
use std::iter::once;

/// Layer: all input weights + bias for all neurons in layer + activation function
/// The Propagation calculation is done in the same Order
#[derive(Debug, Clone)]
pub struct Layer {
    weights: Matrix<f64>,
    bias: LayerBias,
    activation_function: ActivationFn,
}

impl Layer {
    impl_getter! { pub get_weights -> weights: &Matrix<f64> }
    impl_getter! { pub get_weights_mut -> weights: &mut Matrix<f64> }
    impl_getter! { pub get_bias -> bias: &LayerBias }
    impl_getter! { pub get_bias_mut -> bias: &mut LayerBias }
    impl_getter! { pub get_activation_function -> activation_function: &ActivationFn }

    pub fn new(weights: Matrix<f64>, bias: LayerBias, activation_function: ActivationFn) -> Self {
        assert_eq!(
            weights.get_height(),
            bias.get_neuron_count(),
            "Weights and Bias don't have matching neuron counts."
        );
        Self {
            weights,
            bias,
            activation_function,
        }
    }

    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter(
        inputs: usize,
        neurons: usize,
        mut iter: impl Iterator<Item = f64>,
        acti_fn: ActivationFn,
    ) -> Layer {
        let weights = Matrix::from_iter(inputs, neurons, &mut iter);
        let bias = LayerBias::from_iter(neurons, iter);
        Layer::new(weights, bias, acti_fn)
    }

    pub fn get_neuron_count(&self) -> usize {
        self.weights.get_height()
    }

    pub fn get_input_count(&self) -> usize {
        self.weights.get_width()
    }

    /// An Input layer doesn't change the input, but still multiplies by the identity matrix and
    /// uses the identity activation function. It might be a good idea to skip the Input layer to
    /// reduce calculations.
    pub fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .add_entries(self.bias.get_vec())
            .into_iter()
            .map(self.activation_function)
            .collect()
    }

    /// like [`Layer::calculate`] but also calculate the derivatives of the activation function
    pub fn training_calculate(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        (&self.weights * inputs)
            .add_entries(self.bias.get_vec())
            .into_iter()
            .map(|z| {
                (
                    self.activation_function.calculate(z),
                    self.activation_function.derivative(z),
                )
            })
            .unzip()
    }

    pub fn iter_neurons(&self) -> impl Iterator<Item = (&Vec<f64>, &f64)> {
        self.weights.iter_rows().zip(self.bias.iter())
    }

    pub fn iter_mut_neurons(&mut self) -> impl Iterator<Item = (&mut Vec<f64>, &mut f64)> {
        self.weights.iter_rows_mut().zip(self.bias.iter_mut())
    }

    pub(crate) fn backpropagation(
        &self,
        input: &Vec<f64>,
        derivative_output: Vec<f64>,
        output_grad: OutputGradient,
        gradient: &mut GradientLayer,
    ) -> InputGradient {
        let input_count = self.get_input_count();
        let neuron_count = self.get_neuron_count();
        assert_eq!(input.len(), input_count);
        assert_eq!(derivative_output.len(), neuron_count);
        assert_eq!(output_grad.len(), neuron_count);

        let mut inputs_grad = vec![0.0; input_count];

        let mul = |(a, b): (f64, f64)| a * b;
        let weights_sums = derivative_output.into_iter().zip(output_grad).map(mul);

        for (((weights, _), (weights_grad, bias_grad)), weighted_sum_grad) in self
            .iter_neurons()
            .zip(gradient.iter_mut_neurons())
            .zip(weights_sums)
        {
            *bias_grad += weighted_sum_grad;

            for (weight_grad, input) in weights_grad.iter_mut().zip(input) {
                *weight_grad += input * weighted_sum_grad;
            }

            for (input_grad, weight) in inputs_grad.iter_mut().zip(weights) {
                *input_grad += weight * weighted_sum_grad;
            }
        }

        inputs_grad
    }

    pub fn init_zero_gradient(&self) -> GradientLayer {
        let (width, height) = self.weights.get_dimensions();
        let bias_change = self.bias.clone_with_zeros();
        GradientLayer::new(Matrix::with_zeros(width, height), bias_change)
    }
}

impl_IterParams! {Layer: weights, bias }

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self.get_input_count();
        let outputs = self.get_neuron_count();
        write!(
            f,
            "Layer ({inputs} -> {outputs}; {}):",
            self.activation_function
        )?;
        let bias_header = "Biases:".to_string();
        let bias_str_iter = once(bias_header).chain(self.bias.iter().map(ToString::to_string));
        let bias_column_width = bias_str_iter.clone().map(|s| s.len()).max().unwrap_or(0);
        let mut bias_lines = bias_str_iter.map(|s| format!("{s:^bias_column_width$}"));
        for l in self.weights.to_string_with_title("Weights:")?.lines() {
            write!(f, "\n{} {}", l, bias_lines.next().unwrap_or_default())?;
        }
        Ok(())
    }
}
