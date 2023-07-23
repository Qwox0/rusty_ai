use crate::prelude::*;
use std::marker::PhantomData;

// Dimension Markers
pub struct NoDim;
pub struct In<const IN: usize>;

// Layer parts Markers
pub struct NoLayerParts;
pub struct LayerParts {
    weights: Matrix<f64>,
    bias: LayerBias,
}

// Optimizer Markers
pub struct NoOptimizer;
pub struct HasOptimizer(Optimizer);

// Rng Markers
pub struct Seed(Option<u64>);

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LP, RNG> {
    layers: Vec<Layer>,
    layer_parts: LP,
    error_function: Option<ErrorFunction>,

    input_dim: PhantomData<IN>,

    // for generation
    rng: RNG,
    default_activation_function: ActivationFn,
}

impl Default for NeuralNetworkBuilder<NoDim, NoLayerParts, Seed> {
    fn default() -> Self {
        NeuralNetworkBuilder {
            layers: vec![],
            layer_parts: NoLayerParts,
            error_function: None,
            input_dim: PhantomData,
            rng: Seed(None),
            default_activation_function: ActivationFn::default(),
        }
    }
}

impl NeuralNetworkBuilder<NoDim, NoLayerParts, Seed> {
    pub fn rng_seed(mut self, seed: u64) -> Self {
        let _ = self.rng.0.insert(seed);
        self
    }

    pub fn input<const N: usize>(self) -> NeuralNetworkBuilder<In<N>, NoLayerParts, RngWrapper> {
        let rng = RngWrapper::new(self.rng.0);
        NeuralNetworkBuilder { input_dim: PhantomData, rng, ..self }
    }
}

impl<IN, LP, RNG> NeuralNetworkBuilder<IN, LP, RNG> {
    pub fn error_function(mut self, error_function: ErrorFunction) -> Self {
        let _ = self.error_function.insert(error_function);
        self
    }

    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

impl<const IN: usize, LP, RNG> NeuralNetworkBuilder<In<IN>, LP, RNG> {
    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }
}

pub type BuilderWithoutParts<const IN: usize> =
    NeuralNetworkBuilder<In<IN>, NoLayerParts, RngWrapper>;
pub type BuilderWithParts<const IN: usize> = NeuralNetworkBuilder<In<IN>, LayerParts, RngWrapper>;

/// This ensures a consistent interface between [`BuilderWithoutParts`] and [`BuilderWithParts`].
pub trait BuildLayer<const IN: usize> {
    fn _layer(self, layer: Layer) -> BuilderWithoutParts<IN>;

    fn layer(
        self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithParts<IN>;

    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT>;

    fn to_trainable_builder<const OUT: usize>(self) -> TrainableNeuralNetworkBuilder<IN, OUT>;
}

impl<const IN: usize> BuildLayer<IN> for BuilderWithoutParts<IN> {
    /// Add a [`Layer`] to the NeuralNetwork.
    fn _layer(mut self, layer: Layer) -> Self {
        assert_eq!(
            layer.get_input_count(),
            self.last_neuron_count(),
            "input count doesn't match output count of previous layer"
        );
        self.layers.push(layer);
        self
    }

    /// Use [`Initializer`] to add a new [`Layer`] to the NeuralNetwork.
    fn layer(
        mut self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithParts<IN> {
        let input_count = self.last_neuron_count();
        let weights = weights_init.init_weights(&mut self.rng, input_count, neurons);
        let bias = bias_init.init_bias(&mut self.rng, input_count, neurons);
        let layer_parts = LayerParts { weights, bias };
        NeuralNetworkBuilder { layer_parts, ..self }
    }

    /// Builds [`NeuralNetwork`].
    ///
    /// # Panics
    ///
    /// Panics if `OUT` doesn't match the the neuron count of the last layer.
    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT> {
        assert_eq!(self.last_neuron_count(), OUT);
        NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default())
    }

    /// Alias for `.build().to_trainable_builder()`
    ///
    /// # Panics
    ///
    /// See `NeuralNetworkBuilder::build`
    fn to_trainable_builder<const OUT: usize>(self) -> TrainableNeuralNetworkBuilder<IN, OUT> {
        self.build().to_trainable_builder()
    }
}

impl<const IN: usize> BuilderWithParts<IN> {
    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, activation_function: ActivationFn) -> BuilderWithoutParts<IN> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, activation_function);
        NeuralNetworkBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    pub fn use_default_activation_function(self) -> BuilderWithoutParts<IN> {
        let default = self.default_activation_function;
        self.activation_function(default)
    }
}

impl<const IN: usize> BuildLayer<IN> for BuilderWithParts<IN> {
    /// Add a [`Layer`] to the NeuralNetwork.
    /// This uses the `self.default_activation_function` for the previously defined layer.
    fn _layer(self, layer: Layer) -> BuilderWithoutParts<IN> {
        self.use_default_activation_function()._layer(layer)
    }

    /// Use [`Initializer`] to add a new [`Layer`] to the NeuralNetwork.
    /// This uses the `self.default_activation_function` for the previously defined layer.
    fn layer(
        self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> Self {
        self.use_default_activation_function().layer(neurons, weights_init, bias_init)
    }

    /// Builds [`NeuralNetwork`].
    /// This uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// # Panics
    ///
    /// Panics if `OUT` doesn't match the the neuron count of the last layer.
    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT> {
        self.use_default_activation_function().build()
    }

    /// Alias for `.build().to_trainable_builder()`
    /// This uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// # Panics
    ///
    /// See `NeuralNetworkBuilder::build`
    fn to_trainable_builder<const OUT: usize>(self) -> TrainableNeuralNetworkBuilder<IN, OUT> {
        self.use_default_activation_function().to_trainable_builder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let ai = NeuralNetworkBuilder::default().input().build::<2>();
        let input = [1.0, 2.0];
        let prop = ai.propagate(&input).0;

        assert_eq!(prop, input);
    }
}
