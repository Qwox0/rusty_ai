use crate::activation_function::ActivationFunction::*;
use crate::layer::{IsLayer, Layer};
use std::iter::once;

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_neurons: &[usize]) -> NeuralNetwork {
        assert!(layer_neurons.len() >= 2);
        NeuralNetwork {
            layers: layer_neurons
                .windows(2)
                .map(|neurons| Layer::new(neurons[0], neurons[1], ReLU2))
                .chain(once(Layer::new_output(
                    *layer_neurons.last().expect("last element exists"),
                    Sigmoid,
                )))
                .collect::<Vec<_>>(),
        }
    }

    pub fn calculate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs.clone(), |acc, layer| layer.calculate(acc))
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_count = self
            .layers
            .get(0)
            .map(|l| l.get_neuron_count())
            .unwrap_or(0);
        write!(
            f,
            "{} Input{}\n{}",
            input_count,
            if input_count == 1 { "" } else { "s" },
            self.layers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}
