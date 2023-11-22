//! # Layer module

#[allow(unused_imports)]
use crate::NeuralNetwork;
use crate::{
    bias::LayerBias,
    gradient::aliases::{InputGradient, OutputGradient},
    traits::default_params_chain,
    util::EntryAdd,
    *,
};
use matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::iter::once;

/// Layer: all input weights + bias for all neurons in layer + activation
/// function The Propagation calculation is done in the same Order
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    weights: Matrix<f64>,
    bias: LayerBias,
    activation_function: ActivationFn,
}

impl Layer {
    /// Creates a new layer of a [`NeuralNetwork`].
    ///
    /// # Panics
    ///
    /// Panics if the height of `weights` doesn't match the length of `bias`.
    pub fn new(weights: Matrix<f64>, bias: LayerBias, activation_function: ActivationFn) -> Self {
        assert_eq!(
            weights.get_height(),
            bias.get_neuron_count(),
            "Weights and Bias don't have matching neuron counts."
        );
        Self { weights, bias, activation_function }
    }

    /// Returns the [`ActivationFn`] of the Layer.
    pub fn get_activation_function(&self) -> &ActivationFn {
        &self.activation_function
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

    /// Returns the number of neurons in the layer which is equal to the number of outputs.
    pub fn get_neuron_count(&self) -> usize {
        self.weights.get_height()
    }

    /// Returns the number of layer inputs.
    pub fn get_input_count(&self) -> usize {
        self.weights.get_width()
    }

    /// # Panics
    ///
    /// Panics if the length of `inputs` doesn't match the input dimension.
    pub fn propagate(&self, inputs: &[f64]) -> Vec<f64> {
        let weighted_sums = (&self.weights * inputs).add_entries(&self.bias);
        self.activation_function.propagate(weighted_sums)
    }

    /// Returns an [`Iterator`] over the inputs and biases of the layer neurons.
    pub fn iter_neurons(&self) -> impl Iterator<Item = (&Vec<f64>, &f64)> {
        self.weights.iter_rows().zip(&self.bias)
    }

    /// Returns an [`Iterator`] over the inputs and biases of the layer neurons.
    pub fn iter_mut_neurons(&mut self) -> impl Iterator<Item = (&mut Vec<f64>, &mut f64)> {
        self.weights.iter_rows_mut().zip(&mut self.bias)
    }

    pub(crate) fn backpropagate(
        &self,
        input: &[f64],
        output: &[f64],
        output_gradient: OutputGradient,
        gradient: &mut GradientLayer,
    ) -> InputGradient {
        let input_count = self.get_input_count();
        let neuron_count = self.get_neuron_count();
        assert_eq!(input.len(), input_count);
        assert_eq!(output.len(), neuron_count);
        assert_eq!(output_gradient.len(), neuron_count);

        let mut inputs_grad = vec![0.0; input_count];

        let weighted_sums_grad = self.activation_function.backpropagate(output_gradient, output);

        self.iter_neurons()
            .zip(gradient.iter_mut_neurons())
            .zip(weighted_sums_grad)
            .for_each(|(((weights, _), (weights_grad, bias_grad)), weighted_sum_grad)| {
                *bias_grad += weighted_sum_grad;

                for (weight_derivative, input) in weights_grad.iter_mut().zip(input) {
                    *weight_derivative += input * weighted_sum_grad;
                }

                for (input_derivative, weight) in inputs_grad.iter_mut().zip(weights) {
                    *input_derivative += weight * weighted_sum_grad;
                }
            });
        inputs_grad
    }

    /// Creates a [`GradientLayer`] with the same dimensions as `self` and every element
    /// initialized to `0.0`
    pub fn init_zero_gradient(&self) -> GradientLayer {
        let (width, height) = self.weights.get_dimensions();
        let bias_grad = self.bias.clone_with_zeros();
        GradientLayer::new(Matrix::with_zeros(width, height), bias_grad)
    }
}

impl ParamsIter for Layer {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a f64> {
        default_params_chain(&self.weights, &self.bias)
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut f64> {
        default_params_chain(&mut self.weights, &mut self.bias)
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self.get_input_count();
        let outputs = self.get_neuron_count();
        write!(f, "Layer ({inputs} -> {outputs}; {}):", self.activation_function)?;
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
