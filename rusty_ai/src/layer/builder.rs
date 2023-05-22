use rand::{distributions::Uniform, Rng};

use super::{Layer, LayerBias};
use crate::prelude::{ActivationFn, Matrix};

pub trait LayerOrLayerBuilder {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer;
}

// Weights Markers
pub struct Neurons(usize);
pub struct LayerDimensions {
    inputs: usize,
    neurons: usize,
}
pub struct WeightsInitialized(Matrix<f64>);

pub trait BuildWeights {
    fn build_weights(self) -> Matrix<f64>;
}
impl BuildWeights for LayerDimensions {
    fn build_weights(self) -> Matrix<f64> {
        todo!()
    }
}
impl BuildWeights for WeightsInitialized {
    fn build_weights(self) -> Matrix<f64> {
        self.0
    }
}

#[derive(Debug, Clone)]
enum BiasMarker {
    OnePerNeuron { neurons: usize },
    Initialized(LayerBias),
}

impl BiasMarker {
    fn build_bias(self) -> LayerBias {
        match self {
            BiasMarker::OnePerNeuron { neurons } => LayerBias::OnePerNeuron(
                rand::thread_rng()
                    .sample_iter(Uniform::from(0.0..1.0))
                    .take(neurons)
                    .collect(),
            ),
            BiasMarker::Initialized(bias) => bias,
        }
    }
}

const DEFAULT_ACTIVATION_FN: ActivationFn = ActivationFn::default_relu();

#[derive(Debug, Clone)]
pub struct LayerBuilder<W> {
    weights: W,
    bias: BiasMarker,
    activation_function: Option<ActivationFn>,
}

impl LayerBuilder<Neurons> {
    pub fn new(neurons: usize) -> LayerBuilder<Neurons> {
        LayerBuilder {
            weights: Neurons(neurons),
            bias: BiasMarker::OnePerNeuron { neurons },
            activation_function: None,
        }
    }
}

impl LayerBuilder<Neurons> {
    pub fn inputs(self, inputs: usize) -> LayerBuilder<LayerDimensions> {
        let Neurons(neurons) = self.weights;
        let weights = LayerDimensions { inputs, neurons };
        LayerBuilder { weights, ..self }
    }

    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized> {
        assert_eq!(self.weights.0, weights.get_height());
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<LayerDimensions> {
    pub fn with_inputs(inputs: usize, neurons: usize) -> LayerBuilder<LayerDimensions> {
        LayerBuilder::new(neurons).inputs(inputs)
    }

    fn get_dimensions(&self) -> (usize, usize) {
        (self.weights.inputs, self.weights.neurons)
    }

    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized> {
        assert_eq!(self.get_dimensions(), weights.get_dimensions());
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<WeightsInitialized> {
    pub fn with_weights(weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized> {
        LayerBuilder::new(0).weights_unchecked(weights)
    }
}

impl<W> LayerBuilder<W> {
    /// initializes `weights` to a specific value. This ignores any previously set layer dimensions.
    pub fn weights_unchecked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized> {
        let weights = WeightsInitialized(weights);
        LayerBuilder { weights, ..self }
    }

    pub fn bias(mut self, bias: LayerBias) -> LayerBuilder<W> {
        self.bias = BiasMarker::Initialized(bias);
        self
    }

    pub fn activation_function(mut self, act_fn: ActivationFn) -> Self {
        self.activation_function = Some(act_fn);
        self
    }
}

impl<W: BuildWeights> LayerBuilder<W> {
    pub fn build(self) -> Layer {
        Layer {
            weights: self.weights.build_weights(),
            bias: self.bias.build_bias(),
            activation_function: self.activation_function.unwrap_or(DEFAULT_ACTIVATION_FN),
        }
    }
}

impl LayerOrLayerBuilder for LayerBuilder<Neurons> {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        self.inputs(inputs).build()
    }
}

impl<W: BuildWeights> LayerOrLayerBuilder for LayerBuilder<W> {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        let layer = self.build();
        assert_eq!(inputs, layer.get_input_count());
        layer
    }
}

fn test() {
    let builder = LayerBuilder::new(5).inputs(1).build();
}
