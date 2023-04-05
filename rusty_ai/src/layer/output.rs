use super::IsLayer;
#[allow(unused_imports)]
use crate::activation_function::ActivationFunction;

/// An [`ActivationFunction`] is used to calculate the vector for the next layer.
/// => OutputLayers don't have an [`ActivationFunction`]
#[derive(Debug)]
pub struct OutputLayer {
    neuron_count: usize,
}

impl OutputLayer {
    pub fn new(neuron_count: usize) -> OutputLayer {
        OutputLayer { neuron_count }
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
