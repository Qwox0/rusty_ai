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

// Bias Markers
pub struct OnePerNeuron(usize);
pub struct BiasInitialized(LayerBias);

pub trait BuildBias {
    fn build_bias(self) -> LayerBias;
}
impl BuildBias for OnePerNeuron {
    fn build_bias(self) -> LayerBias {
        let vec = rand::thread_rng()
            .sample_iter(Uniform::from(0.0..1.0))
            .take(self.0)
            .collect();
        LayerBias::OnePerNeuron(vec)
    }
}
impl BuildBias for BiasInitialized {
    fn build_bias(self) -> LayerBias {
        self.0
    }
}

const DEFAULT_ACTIVATION_FN: ActivationFn = ActivationFn::default_relu();

#[derive(Debug, Clone)]
pub struct LayerBuilder<W, B> {
    weights: W,
    bias: B,
    activation_function: Option<ActivationFn>,
}

impl LayerBuilder<Neurons, OnePerNeuron> {
    pub fn new(neurons: usize) -> LayerBuilder<Neurons, OnePerNeuron> {
        LayerBuilder {
            weights: Neurons(neurons),
            bias: OnePerNeuron(neurons),
            activation_function: None,
        }
    }
}

impl<B> LayerBuilder<Neurons, B> {
    pub fn inputs(self, inputs: usize) -> LayerBuilder<LayerDimensions, B> {
        let Neurons(neurons) = self.weights;
        let weights = LayerDimensions { inputs, neurons };
        LayerBuilder { weights, ..self }
    }

    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        assert_eq!(self.weights.0, weights.get_height());
        self.weights_unchecked(weights)
    }
}

impl<B> LayerBuilder<LayerDimensions, B> {
    pub fn with_inputs(
        inputs: usize,
        neurons: usize,
    ) -> LayerBuilder<LayerDimensions, OnePerNeuron> {
        LayerBuilder::new(neurons).inputs(inputs)
    }

    fn get_dimensions(&self) -> (usize, usize) {
        (self.weights.inputs, self.weights.neurons)
    }

    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        assert_eq!(self.get_dimensions(), weights.get_dimensions());
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<WeightsInitialized, OnePerNeuron> {
    pub fn with_weights(weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, OnePerNeuron> {
        LayerBuilder::new(0).weights_unchecked(weights)
    }
}

impl<W, B> LayerBuilder<W, B> {
    /// initializes `weights` to a specific value. This ignores any previously set layer dimensions.
    pub fn weights_unchecked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        let weights = WeightsInitialized(weights);
        LayerBuilder { weights, ..self }
    }

    pub fn bias(self, bias: LayerBias) -> LayerBuilder<W, BiasInitialized> {
        let bias = BiasInitialized(bias);
        LayerBuilder { bias, ..self }
    }

    pub fn activation_function(mut self, act_fn: ActivationFn) -> Self {
        self.activation_function = Some(act_fn);
        self
    }
}

impl<W: BuildWeights, B: BuildBias> LayerBuilder<W, B> {
    pub fn build(self) -> Layer {
        Layer {
            weights: self.weights.build_weights(),
            bias: self.bias.build_bias(),
            activation_function: self.activation_function.unwrap_or(DEFAULT_ACTIVATION_FN),
        }
    }
}

impl<B: BuildBias> LayerOrLayerBuilder for LayerBuilder<Neurons, B> {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        self.inputs(inputs).build()
    }
}

impl<W: BuildWeights, B: BuildBias> LayerOrLayerBuilder for LayerBuilder<W, B> {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        let layer = self.build();
        assert_eq!(inputs, layer.get_input_count());
        layer
    }
}

fn test() {
    let builder = LayerBuilder::new(5).inputs(1).build();
}
