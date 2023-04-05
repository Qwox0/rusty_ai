mod normal;
mod output;

pub use normal::NormalLayer;
pub use output::OutputLayer;

use crate::activation_function::ActivationFunction;

pub trait IsLayer: std::fmt::Display {
    fn get_neuron_count(&self) -> usize;
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64>;

}

#[derive(Debug)]
pub enum Layer {
    //Input(NormalLayer),
    //Hidden(NormalLayer),
    Normal(NormalLayer),
    Output(OutputLayer),
}

impl Layer {
    pub fn new(neuron_count: usize, next_neuron_count: usize, activation_function: ActivationFunction) -> Layer {
        Layer::Normal(NormalLayer::new(neuron_count, next_neuron_count, activation_function))
    }

    pub fn new_output(neuron_count: usize, activation_function: ActivationFunction) -> Layer {
        Layer::Output(OutputLayer::new(neuron_count, activation_function))
    }
}

impl IsLayer for Layer {
    fn get_neuron_count(&self) -> usize {
        match self {
            Layer::Normal(l) => l.get_neuron_count(),
            Layer::Output(l) => l.get_neuron_count(),
        }
    }

    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        match self {
            Layer::Normal(l) => l.calculate(inputs),
            Layer::Output(l) => l.calculate(inputs),
        }
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Normal(l) => write!(f, "{}", l),
            Layer::Output(l) => write!(f, "{}", l),
        }
    }
}

