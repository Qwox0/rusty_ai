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

// Rng Markers
pub struct Seed(Option<u64>);

/// Builder
///
/// # Generics
///
/// `IN`: input dimension (not set yet or const usize)
/// `LP`: layer parts (contains weights and bias til next activation function is set)
/// `RNG`: rng type (seeded or not)
#[derive(Debug)]
pub struct NNBuilder<IN, LP, RNG> {
    layers: Vec<Layer>,
    layer_parts: LP,

    input_dim: PhantomData<IN>,

    // for generation
    default_activation_function: ActivationFn,
    rng: RNG,
}

impl Default for NNBuilder<NoDim, NoLayerParts, Seed> {
    fn default() -> Self {
        NNBuilder {
            layers: vec![],
            layer_parts: NoLayerParts,
            input_dim: PhantomData,
            default_activation_function: ActivationFn::default(),
            rng: Seed(None),
        }
    }
}

impl NNBuilder<NoDim, NoLayerParts, Seed> {
    pub fn rng_seed(mut self, seed: u64) -> Self {
        let _ = self.rng.0.insert(seed);
        self
    }

    pub fn input<const N: usize>(self) -> NNBuilder<In<N>, NoLayerParts, RngWrapper> {
        let rng = RngWrapper::new(self.rng.0);
        NNBuilder { input_dim: PhantomData, rng, ..self }
    }
}

impl<IN, LP, RNG> NNBuilder<IN, LP, RNG> {
    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

impl<const IN: usize, LP, RNG> NNBuilder<In<IN>, LP, RNG> {
    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }
}

pub type BuilderWithoutParts<const IN: usize> = NNBuilder<In<IN>, NoLayerParts, RngWrapper>;
pub type BuilderWithParts<const IN: usize> = NNBuilder<In<IN>, LayerParts, RngWrapper>;

/// This ensures a consistent interface between [`BuilderWithoutParts`] and [`BuilderWithParts`].
pub trait BuildLayer<const IN: usize>: Sized {
    fn _layer(self, layer: Layer) -> BuilderWithoutParts<IN>;

    fn layer(
        self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithParts<IN>;

    fn layers(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
        activation_function: ActivationFn,
    ) -> BuilderWithoutParts<IN>;

    fn layers_default(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithoutParts<IN>;

    /// Create a new [`Layer`] from the given `weights` and `bias` and add it to the NeuralNetwork.
    /// See `layer` method.
    fn layer_from_parameters(self, weights: Matrix<f64>, bias: LayerBias) -> BuilderWithParts<IN> {
        use Initializer::Initialized;
        self.layer(weights.get_height(), Initialized(weights), Initialized(bias))
    }

    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT>;

    fn to_trainable_builder<const OUT: usize>(
        self,
    ) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer>;
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
        NNBuilder { layer_parts, ..self }
    }

    /// Use the same [`Initializer`] to add multiple new [`Layer`]s to the NeuralNetwork.
    /// Every new layer gets `activation_function`.
    /// This method calls `clone` on `weights_init` and `bias_init`.
    /// See `layer` method.
    fn layers(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
        activation_function: ActivationFn,
    ) -> BuilderWithoutParts<IN> {
        neurons.into_iter().fold(self, |builder: Self, neurons: &usize| {
            builder
                .layer(*neurons, weights_init.clone(), bias_init.clone())
                .activation_function(activation_function)
        })
    }

    /// similar to `layers` but uses the default activation function for every layer.
    fn layers_default(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithoutParts<IN> {
        let default = self.default_activation_function;
        self.layers(neurons, weights_init, bias_init, default)
    }

    /// Builds [`NeuralNetwork`].
    ///
    /// # Panics
    ///
    /// Panics if `OUT` doesn't match the the neuron count of the last layer.
    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT> {
        assert_eq!(self.last_neuron_count(), OUT);
        NeuralNetwork::new(self.layers)
    }

    /// Alias for `.build().to_trainable_builder()`
    ///
    /// # Panics
    ///
    /// See `NeuralNetworkBuilder::build`
    fn to_trainable_builder<const OUT: usize>(
        self,
    ) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
        self.build().to_trainable_builder()
    }
}

macro_rules! activation_function {
    ( $( $fn_name:ident -> $variant:ident $( { $($arg:ident : $ty:ty),+ } )? : $variant_str:expr );+ ) => {
        $(
            #[doc = "Sets the `"]
            #[doc = $variant_str]
            #[doc = "` activation function for the previously defined layer."]
            pub fn $fn_name(self $(, $($arg : $ty),+)? ) -> BuilderWithoutParts<IN> {
                self.activation_function(ActivationFn::$variant $({ $($arg),+ })?)
            }
         )+
    };
}

impl<const IN: usize> BuilderWithParts<IN> {
    activation_function! {
        identity -> Identity : "Identity" ;
        relu -> ReLU : "ReLU" ;
        leaky_relu -> LeakyReLU { leak_rate: f64 } : "LeakyReLU" ;
        sigmoid -> Sigmoid : "Sigmoid" ;
        softmax -> Softmax : "Softmax" ;
        log_softmax -> LogSoftmax : "LogSoftmax"
    }

    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, af: ActivationFn) -> BuilderWithoutParts<IN> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, af);
        NNBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// This function gets called automatically if no activation function is provided.
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

    /// Use the same [`Initializer`] to add multiple new [`Layer`]s to the NeuralNetwork.
    /// This uses the `self.default_activation_function` for the previously defined layer. Every
    /// new layer gets `activation_function`. This method calls `clone`
    /// on `weights_init` and `bias_init`. See `layer` method.
    fn layers(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
        activation_function: ActivationFn,
    ) -> BuilderWithoutParts<IN> {
        self.use_default_activation_function().layers(
            neurons,
            weights_init,
            bias_init,
            activation_function,
        )
    }

    /// similar to `layers` but uses the default activation function for the previous layer and
    /// every new layer.
    fn layers_default(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithoutParts<IN> {
        self.use_default_activation_function().layers_default(neurons, weights_init, bias_init)
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
    fn to_trainable_builder<const OUT: usize>(
        self,
    ) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
        self.use_default_activation_function().to_trainable_builder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let ai = NNBuilder::default().input().build::<2>();
        let input = [1.0, 2.0];
        let prop = ai.propagate(&input).0;

        assert_eq!(prop, input);
    }
}
