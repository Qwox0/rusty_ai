use crate::{
    activation_function::ActivationFunction::{self, *},
    matrix::Matrix,
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

impl Layer {
    fn new(
        layer_type: LayerType,
        weights: Matrix<f64>,
        bias: f64,
        activation_function: ActivationFunction,
    ) -> Layer {
        Layer {
            layer_type,
            weights,
            bias,
            activation_function,
        }
    }

    pub fn new_input(neurons: usize) -> Layer {
        Layer::new(Input, Matrix::identity(neurons), 0.0, Identity)
    }

    pub fn new_hidden(inputs: usize, neurons: usize, acti_func: ActivationFunction) -> Layer {
        Layer::new(
            Hidden,
            Matrix::new_random(inputs, neurons),
            rand::random(),
            ReLU2,
        )
    }

    pub fn new_output(inputs: usize, neurons: usize, acti_func: ActivationFunction) -> Layer {
        Layer::new(
            Output,
            Matrix::new_random(inputs, neurons),
            rand::random(),
            ReLU2,
        )
    }

    pub fn get_input_count(&self) -> usize {
        *self.weights.get_width()
    }

    pub fn get_neuron_count(&self) -> usize {
        *self.weights.get_height()
    }

    /// Input layer doesn't change the input, but multiplies by the identity matrix and uses the
    /// identity activation function. It might be good to skip the Input layer.
    pub fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .apply(|x| x + self.bias)
            .apply(self.activation_function)
    }
    pub fn calculate2(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut res = &self.weights * inputs;
        for x in res.iter_mut() {
            *x = (self.activation_function)(*x + self.bias)
        }
        res
    }
    pub fn calculate3(&self, inputs: Vec<f64>) -> Vec<f64> {
        (&self.weights * inputs)
            .into_iter()
            .map(|x| x + self.bias)
            .map(&self.activation_function)
            .collect()
    }
}

// -----------------------------
pub trait Apply: Sized {
    type Elem;
    fn apply_mut(self, f: impl FnMut(&mut Self::Elem)) -> Self;
    fn apply(self, f: impl FnMut(Self::Elem) -> Self::Elem) -> Self {
        self.apply_mut(|a| *a = f(*a))
    }
}

impl<T> Apply for Vec<T> {
    type Elem = T;

    fn apply_mut(mut self, f: impl FnMut(&mut Self::Elem)) -> Self {
        self.iter_mut().for_each(f);
        self
    }

    /*
    fn apply(mut self, f: impl FnMut(Self::Elem) -> Self::Elem) -> Self {
        for e in self.iter_mut() {
            *e = f(*e)
        }
        self
    }
        */
}
// -----------------------------

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plural_s = |x: usize| if x == 1 { "" } else { "s" };
        if let Input = self.layer_type {
            let input_count = self.get_input_count();
            write!(f, "{} Input{}", input_count, plural_s(input_count))?;
        }
        write!(
            f,
            "{} Bias: {}; {}",
            self.weights, self.bias, self.activation_function
        )?;
        if let Output = self.layer_type {
            let output_count = self.get_neuron_count();
            write!(f, "{} Output{}", output_count, plural_s(output_count))?;
        }
        Ok(())
    }
}
