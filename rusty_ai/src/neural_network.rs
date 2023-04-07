use crate::layer::Layer;

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

#[allow(unused)]
use crate::builder::NeuralNetworkBuilder;

impl NeuralNetwork {
    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(layers: Vec<Layer>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }

    pub fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .skip(1) // first layer is always an input layer which doesn't mutate the input
            .fold(inputs, |acc, layer| layer.calculate(acc))
    }

    pub fn calculate_ref(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.calculate(inputs.clone())
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.layers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}
