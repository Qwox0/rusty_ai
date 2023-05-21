use super::{Layer, LayerBias};
use crate::prelude::{ActivationFn, Matrix};

// Markers
pub struct InputsMissing;

/// # Fields
/// `inputs`: required
/// `neurons`: required
/// `bias`: not required (default: `ActivationFn::default_relu()`)
/// `activation_function`: not required (default: `ActivationFn::default_relu()`)
pub struct LayerBuilder<I> {
    inputs: I,
    neurons: usize,
    weights: Option<Matrix<f64>>,
    bias: Option<LayerBias>,
    activation_function: Option<ActivationFn>,
}

impl LayerBuilder<InputsMissing> {
    pub fn new(neurons: usize) -> LayerBuilder<InputsMissing> {
        LayerBuilder {
            inputs: InputsMissing,
            neurons,
            weights: None,
            bias: None,
            activation_function: None,
        }
    }

    pub fn inputs(self, inputs: usize) -> LayerBuilder<usize> {
        LayerBuilder { inputs, ..self }
    }

    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<usize> {
        assert_eq!(self.neurons, weights.get_height());
        LayerBuilder {
            inputs: weights.get_width(),
            weights: Some(weights),
            ..self
        }
    }
}

impl LayerBuilder<usize> {
    pub fn with_inputs(inputs: usize, neurons: usize) -> LayerBuilder<usize> {
        LayerBuilder::new(neurons).inputs(inputs)
    }

    pub fn with_weights(weights: Matrix<f64>) -> LayerBuilder<usize> {
        LayerBuilder::new(0).weights(weights)
    }

    fn get_dimensions(&self) -> (usize, usize) {
        (self.inputs, self.neurons)
    }

    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<usize> {
        assert_eq!(self.get_dimensions(), weights.get_dimensions());
        LayerBuilder {
            inputs: weights.get_width(),
            weights: Some(weights),
            ..self
        }
    }

    pub fn build(self) -> Layer {
        let activation_function = self
            .activation_function
            .unwrap_or(ActivationFn::default_relu());
        let weights = self.weights.unwrap_or(todo!());
        let bias = self.bias.unwrap_or(todo!());
        Layer {
            weights,
            bias,
            activation_function,
        }
    }
}

impl<I> LayerBuilder<I> {
    pub fn weights(self, weights: Matrix<f64>) -> LayerBuilder<usize> {
        LayerBuilder {
            inputs: weights.get_width(),
            neurons: weights.get_height(),
            weights: Some(weights),
            ..self
        }
    }

    pub fn bias(mut self, bias: LayerBias) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn activation_function(mut self, act_fn: ActivationFn) -> Self {
        self.activation_function = Some(act_fn);
        self
    }
}

fn test() {
    let builder = LayerBuilder::new(5).inputs(1).build();
}
