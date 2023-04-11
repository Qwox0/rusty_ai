use crate::{
    activation_function::ActivationFunction::{self, *},
    matrix::Matrix,
    results::GradientLayer,
    util::{
        macros::{impl_getter, impl_new},
        EntryAdd, EntrySub, ScalarMul,
    },
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
pub(crate) struct Layer {
    layer_type: LayerType,
    weights: Matrix<f64>,
    bias: f64,
    activation_function: ActivationFunction,
}

impl Layer {
    impl_new! { pub(crate) layer_type: LayerType, weights: Matrix<f64>, bias: f64, activation_function: ActivationFunction }
    impl_getter! { get_layer_type -> layer_type: &LayerType }
    impl_getter! { get_weights -> weights: &Matrix<f64> }
    impl_getter! { get_weights_mut -> weights: &mut Matrix<f64> }
    impl_getter! { get_bias -> bias: f64 }
    impl_getter! { get_bias_mut -> bias: &mut f64 }
    impl_getter! { get_activation_function -> activation_function: &ActivationFunction }

    pub(crate) fn new_input(neurons: usize) -> Layer {
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
        self.weights.get_width()
    }

    pub fn get_neuron_count(&self) -> usize {
        self.weights.get_height()
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

    /// like [`Layer::calculate`] but also calculate the derivatives of the activation function
    pub fn training_calculate(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        (&self.weights * inputs)
            .into_iter()
            .map(|x| x + self.bias)
            .map(|z| {
                (
                    self.activation_function.calculate(z),
                    self.activation_function.derivative(z),
                )
            })
            .unzip()
    }

    pub(crate) fn iter_neurons(&self) -> impl Iterator<Item = &Vec<f64>> {
        self.weights.iter_rows()
    }

    pub(crate) fn backpropagation(
        &self,
        layer_inputs: &Vec<f64>,
        layer_outputs: &Vec<f64>,
        derivative_outputs: Vec<f64>,
        expected_outputs: &Vec<f64>,
    ) -> (f64, Matrix<f64>, Vec<f64>) {
        let layer_input_count = self.get_input_count();
        let layer_neuron_count = self.get_neuron_count();
        assert_eq!(
            derivative_outputs.len(),
            layer_neuron_count,
            "derivatives is the wrong size"
        );
        assert_eq!(
            layer_inputs.len(),
            layer_input_count,
            "layer_inputs is the wrong size"
        );
        assert_eq!(
            layer_outputs.len(),
            layer_neuron_count,
            "layer_outputs is the wrong size"
        );
        assert_eq!(
            expected_outputs.len(),
            layer_neuron_count,
            "expected_output is the wrong size"
        );

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
                    let dc_dbias = err * weighted_sum_derivative;
                    // dC/dw_ij      = (o_L_i - e_i) *     f'(z_i) *  o_(L-1)_j
                    let dc_dweights = layer_inputs.clone().mul_scalar(dc_dbias);
                    // dC/do_(L-1)_j = (o_L_i - e_i) *     f'(z_i) *       w_ij ()
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
                    Matrix::new_unchecked(layer_input_count, layer_neuron_count),
                    // sum of partial derivatives with respect to input (next expected output)
                    vec![0.0; layer_input_count],
                ),
                |mut acc, (dc_dbias, dc_dweights, dc_dinputs)| {
                    acc.0 += dc_dbias;
                    acc.1.push_row(dc_dweights);
                    acc.2.add_into(&dc_dinputs);
                    acc
                },
            );
        let bias_derivative_average = res.0 / layer_neuron_count as f64;
        let input_derivative_average = res.2.mul_scalar(1.0 / layer_neuron_count as f64);
        (bias_derivative_average, res.1, input_derivative_average)
    }

    pub(crate) fn backpropagation2(
        &self,
        layer_inputs: &Vec<f64>,
        layer_outputs: &Vec<f64>,
        derivative_outputs: Vec<f64>,
        expected_outputs: &Vec<f64>,
    ) -> (f64, Matrix<f64>, Vec<f64>) {
        let layer_input_count = self.get_input_count();
        let layer_neuron_count = self.get_neuron_count();

        #[cfg(debug_assertions)]
        assert!(self.weights.get(0, 0).unwrap().abs() < 1000000.0);
        #[cfg(debug_assertions)]
        assert!(!self.weights.get(0, 0).unwrap().is_nan());
        assert_eq!(
            derivative_outputs.len(),
            layer_neuron_count,
            "derivatives is the wrong size"
        );
        assert_eq!(
            layer_inputs.len(),
            layer_input_count,
            "layer_inputs is the wrong size"
        );
        assert_eq!(
            layer_outputs.len(),
            layer_neuron_count,
            "layer_outputs is the wrong size"
        );
        assert_eq!(
            expected_outputs.len(),
            layer_neuron_count,
            "expected_output is the wrong size"
        );

        #[cfg(debug_assertions)]
        print!(
            "expected outputs: {:.4?}; {:.4} got: {:.4?}",
            expected_outputs,
            " ".repeat(
                40usize
                    .checked_sub(format!("{expected_outputs:?}").len())
                    .unwrap_or(2)
            ),
            layer_outputs
        );

        let mut err = layer_outputs.clone().sub(expected_outputs); // size: 2

        #[cfg(debug_assertions)]
        println!("   -> ERR: {err:.4?}");

        err.iter_mut()
            .zip(derivative_outputs.iter())
            .for_each(|(x, d)| *x *= d);
        let neuron_change = err;

        let bias_change: f64 = neuron_change.iter().sum(); // size: 1

        let mut weight_changes = Matrix::new_unchecked(layer_input_count, layer_neuron_count); // width: 3, height: 2
        for neuron in neuron_change.iter() {
            weight_changes.push_row(layer_inputs.clone().mul_scalar(*neuron));
        }

        let mut input_changes = vec![0.0; layer_input_count];
        for (weights, change) in self.iter_neurons().zip(neuron_change) {
            input_changes.add_into(weights.clone().mul_scalar(change));
            //input_changes.add_into(weights.clone().mul_scalar(change.signum()));
        }

        #[cfg(debug_assertions)]
        {
            println!("BIAS CHANGE   : {:.7}", bias_change);
            println!(
                "WEIGHTS CHANGE: {:.7?}    (inputs: {:.4?})",
                weight_changes, layer_inputs
            );
            println!(
                "INPUTS CHANGE : {:.7?}    (weights: {:.4?})",
                input_changes, self.weights
            );
        }

        (bias_change, weight_changes, input_changes)
    }

    pub fn init_gradient(&self) -> GradientLayer {
        let (width, height) = self.weights.get_dimensions();
        GradientLayer::new(Matrix::with_zeros(width, height), 0.0)
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
