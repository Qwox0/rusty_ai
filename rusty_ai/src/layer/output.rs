use super::IsLayer;
use crate::activation_function::ActivationFunction;

#[derive(Debug)]
pub struct OutputLayer {
    neuron_count: usize,
    activation_function: ActivationFunction,
}

impl OutputLayer {
    pub fn new(neuron_count: usize, activation_function: ActivationFunction) -> OutputLayer {
        OutputLayer {
            neuron_count,
            activation_function,
        }
    }
}

impl IsLayer for OutputLayer {
    fn get_neuron_count(&self) -> usize {
        self.neuron_count
    }

    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        inputs
    }
}

impl std::fmt::Display for OutputLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output_count = self.get_neuron_count();
        write!(
            f,
            "{} Output{}",
            output_count,
            if output_count == 1 { "" } else { "s" }
        )
    }
}
