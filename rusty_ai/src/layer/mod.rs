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

/*
#[derive(Debug, Clone)]
pub enum LayerType {
    Hidden,
    Output,
}
use LayerType::*;
*/

pub trait IsLayer: std::fmt::Display {
    fn get_neuron_count(&self) -> usize;
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64>;
}

/// Layer: all input weights + bias for all neurons in layer + activation function
/// The Propagation calculation is done in the same Order
#[derive(Debug, Clone)]
pub struct Layer {
    weights: Matrix<f64>,
    bias: LayerBias,
    activation_function: ActivationFn,
}

impl IsLayer for Layer {
    fn get_neuron_count(&self) -> usize {
        self.weights.get_height()
    }

    /// An Input layer doesn't change the input, but still multiplies by the identity matrix and
    /// uses the identity activation function. It might be a good idea to skip the Input layer to
    /// reduce calculations.
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .add_entries_mut(self.bias.get_vec())
            .iter()
            .map(self.activation_function)
            .collect()
    }
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

    pub fn get_input_count(&self) -> usize {
        self.weights.get_width()
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

    /*
    pub(crate) fn backpropagation(
        &self,
        layer_inputs: &Vec<f64>,
        layer_outputs: &Vec<f64>,
        derivative_outputs: Vec<f64>,
        expected_outputs: &Vec<f64>,
    ) -> (f64, Matrix<f64>, Vec<f64>) {
        let layer_input_count = self.get_input_count();
        let layer_neuron_count = self.get_neuron_count();
        assert_eq!(derivative_outputs.len(), layer_neuron_count);
        assert_eq!(layer_inputs.len(), layer_input_count);
        assert_eq!(layer_outputs.len(), layer_neuron_count);
        assert_eq!(expected_outputs.len(), layer_neuron_count);

        // variables: dc_dx = dC/dx = partial derivative of the cost function
        //                            with respect to x.

        let res = self
            .iter_neurons()
            .zip(derivative_outputs.iter())
            .zip(layer_outputs)
            .zip(expected_outputs.iter())
            .map(|(((l, d), lo), eo)| (l, d, lo, eo))
            .map(
                |(neuron_weights, weighted_sum_derivative, output, expected_output)| {
                    let err = output - expected_output;
                    //print!("err: {:?} ", err);

                    // dC/db_L       = (o_L_i - e_i) *     f'(z_i)
                    let dc_dbias = 2.0 * err * weighted_sum_derivative;
                    // dC/dw_ij      = (o_L_i - e_i) *     f'(z_i) *  o_(L-1)_j
                    let dc_dweights = layer_inputs.clone().mul_scalar(dc_dbias);
                    // dC/do_(L-1)_j = (o_L_i - e_i) *     f'(z_i) *       w_ij
                    let dc_dinputs = neuron_weights.clone().mul_scalar(dc_dbias);
                    /*
                    println!(
                        "Bias_d: {}; weights_derivatives: {:?}; inputs_d: {:?}",
                        &dc_dbias, &dc_dweights, &dc_dinputs
                    );
                    */

                    (dc_dbias, dc_dweights, dc_dinputs)
                },
            )
            .fold(
                (
                    // sum of partial derivatives with respect to bias
                    0.0,
                    // partial derivatives with respect to weights: matrix row i
                    // contains the weights change of connections to layer neuron i.
                    Matrix::new_empty(layer_input_count, layer_neuron_count),
                    // sum of partial derivatives with respect to input (next expected output)
                    vec![0.0; layer_input_count],
                ),
                |mut acc, (dc_dbias, dc_dweights, dc_dinputs)| {
                    acc.0 += dc_dbias;
                    acc.1.push_row(dc_dweights);
                    acc.2.add_entries_mut(&dc_dinputs);
                    acc
                },
            );
        let bias_derivative_average = res.0 / layer_neuron_count as f64;
        let input_derivative_average = res.2.mul_scalar(1.0 / layer_neuron_count as f64);
        (bias_derivative_average, res.1, input_derivative_average)
    }
    */

    pub(crate) fn backpropagation2(
        &self,
        derivative_output: Vec<f64>,
        input: Vec<f64>,
        output_gradient: OutputGradient,
        gradient: &mut GradientLayer,
    ) -> InputGradient {
        let input_count = self.get_input_count();
        let neuron_count = self.get_neuron_count();
        assert_eq!(derivative_output.len(), neuron_count);
        assert_eq!(input.len(), input_count);
        assert_eq!(output_gradient.len(), neuron_count);

        let mut input_gradient = vec![0.0; input_count];

        self.iter_neurons()
            .zip(gradient.iter_mut_neurons())
            .zip(derivative_output)
            .zip(output_gradient)
            .for_each(
                |((((weights, _), (weights_grad, bias_grad)), act_derivative), out_grad)| {
                    let weighted_sum_gradient = out_grad * act_derivative;

                    *bias_grad += weighted_sum_gradient;

                    for (weight_grad, input) in weights_grad.iter_mut().zip(&input) {
                        *weight_grad += input * weighted_sum_gradient;
                    }

                    for (weight, input_grad) in weights.iter().zip(&mut input_gradient) {
                        *input_grad += weight * weighted_sum_gradient;
                    }
                },
            );

        input_gradient
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
