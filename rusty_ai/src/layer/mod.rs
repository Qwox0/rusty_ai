mod add_bias;
mod bias;
mod builder;
mod input;

pub use add_bias::*;
pub use bias::*;
pub use builder::*;
pub use input::InputLayer;

use crate::{
    activation_function::ActivationFn,
    gradient::aliases::{
        BiasGradient, InputGradient, OutputGradient, WeightGradient, WeightedSumGradient,
    },
    gradient::layer::GradientLayer,
    matrix::Matrix,
    util::{constructor, impl_getter, EntryAdd, EntryMul, EntrySub, Randomize, ScalarMul},
};

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
    //layer_type: LayerType,
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
            .add_bias_mut(&self.bias)
            .iter()
            .map(self.activation_function)
            .collect()
    }
}

impl Layer {
    constructor! { new -> weights: Matrix<f64>, bias: LayerBias, activation_function: ActivationFn }
    impl_getter! { pub get_weights -> weights: &Matrix<f64> }
    impl_getter! { pub get_weights_mut -> weights: &mut Matrix<f64> }
    impl_getter! { pub get_bias -> bias: &LayerBias }
    impl_getter! { pub get_bias_mut -> bias: &mut LayerBias }
    impl_getter! { pub get_activation_function -> activation_function: &ActivationFn }

    pub fn random_with_bias(
        inputs: usize,
        neurons: usize,
        bias: LayerBias,
        acti_func: ActivationFn,
    ) -> Layer {
        Layer::new(Matrix::new_random(inputs, neurons), bias, acti_func)
    }

    pub fn random(inputs: usize, neurons: usize, acti_func: ActivationFn) -> Layer {
        Layer::random_with_bias(
            inputs,
            neurons,
            LayerBias::new_multiple(vec![0.0; neurons]).randomize_uniform(0.0..1.0),
            acti_func,
        )
    }

    pub fn get_input_count(&self) -> usize {
        self.weights.get_width()
    }

    /// like [`Layer::calculate`] but also calculate the derivatives of the activation function
    pub fn training_calculate(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        (&self.weights * inputs)
            .add_bias_mut(&self.bias)
            .iter()
            .map(|z| {
                (
                    self.activation_function.calculate(*z),
                    self.activation_function.derivative(*z),
                )
            })
            .unzip()
    }

    pub fn iter_neurons(&self) -> impl Iterator<Item = &Vec<f64>> {
        self.weights.iter_rows()
    }

    pub fn iter_mut_neurons(&mut self) -> impl Iterator<Item = &mut Vec<f64>> {
        self.weights.iter_rows_mut()
    }

    #[allow(unused)]
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

    #[allow(unused)]
    pub(crate) fn backpropagation2(
        &self,
        derivative_output: Vec<f64>,
        input: &Vec<f64>,
        output_gradient: OutputGradient,
    ) -> (BiasGradient, WeightGradient, InputGradient) {
        let layer_input_count = self.get_input_count();
        let layer_neuron_count = self.get_neuron_count();
        assert_eq!(derivative_output.len(), layer_neuron_count,);
        assert_eq!(input.len(), layer_input_count,);
        assert_eq!(output_gradient.len(), layer_neuron_count,);

        let weighted_sum_gradient: WeightedSumGradient =
            output_gradient.mul_entries(derivative_output);

        let bias_gradient: BiasGradient = self.bias.new_matching_gradient(&weighted_sum_gradient);

        let mut weight_gradient: WeightGradient =
            Matrix::new_empty(layer_input_count, layer_neuron_count);
        for &neuron in weighted_sum_gradient.iter() {
            weight_gradient.push_row(input.clone().mul_scalar(neuron));
        }

        let mut input_gradient: InputGradient = vec![0.0; layer_input_count];
        for (weights, change) in self.iter_neurons().zip(weighted_sum_gradient) {
            input_gradient.add_entries_mut(weights.clone().mul_scalar(change));
        }

        (bias_gradient, weight_gradient, input_gradient)
    }

    pub fn init_zero_gradient(&self) -> GradientLayer {
        let (width, height) = self.weights.get_dimensions();
        let bias_change = self.bias.clone_with_zeros();
        GradientLayer::new(Matrix::with_zeros(width, height), bias_change)
    }
}

impl Randomize for Layer {
    type Sample = f64;

    fn _randomize_mut(
        &mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) {
        self.weights._randomize_mut(rng, &distr);
        self.bias._randomize_mut(rng, &distr);
    }
}

/*
impl MultiRandom for Layer {
    type Sample;
    type Item;
    type Size;

    fn _random_multiple(
        rng: &mut impl rand::Rng,
        distr: impl rand::prelude::Distribution<Self::Sample>,
        count: Self::Size,
    ) -> Self {
        todo!()
    }
}
*/

impl EntrySub<&GradientLayer> for Layer {
    fn sub_entries_mut(&mut self, rhs: &GradientLayer) -> &mut Self {
        self.weights.sub_entries_mut(&rhs.weight_gradient);
        self.bias.sub_entries_mut(&rhs.bias_gradient);
        self
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Bias: {}; {}",
            self.weights, self.bias, self.activation_function
        )
    }
}

impl LayerOrLayerBuilder for Layer {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        assert_eq!(self.get_input_count(), inputs);
        self
    }
}
