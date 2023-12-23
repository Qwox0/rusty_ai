//! # Neural network builder module

#[allow(unused_imports)]
use crate::trainer::NNTrainer;
use crate::{
    bias::LayerBias,
    layer::Layer,
    matrix::Matrix,
    trainer::markers::{NoLossFunction, NoOptimizer},
    *,
};
use markers::*;
use rand::{
    rngs::{StdRng, ThreadRng},
    SeedableRng,
};
use std::marker::PhantomData;
use trainer::NNTrainerBuilder;

/// Markers uses by [`NNBuilder`].
pub mod markers {
    #[allow(unused_imports)]
    use crate::layer::Layer;
    use crate::{bias::LayerBias, matrix::Matrix};

    /// Marker for an undefined Dimension.
    pub struct NoDim;
    /// Inputs dimension == `IN`
    pub struct In<const IN: usize>;

    /// Builder doesn't contain an unfinished [`Layer`].
    pub struct NoLayerParts;
    /// Builder contains an unfinished [`Layer`].
    pub struct LayerParts {
        pub(super) weights: Matrix<f64>,
        pub(super) bias: LayerBias,
    }

    /// Seed used for RNG.
    pub struct NoRng;
}

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

impl Default for NNBuilder<NoDim, NoLayerParts, NoRng> {
    fn default() -> Self {
        NNBuilder {
            layers: vec![],
            layer_parts: NoLayerParts,
            input_dim: PhantomData,
            default_activation_function: ActivationFn::default(),
            rng: NoRng,
        }
    }
}

impl<DIM, LP, RNG> NNBuilder<DIM, LP, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> NNBuilder<DIM, LP, R> {
        NNBuilder { rng, ..self }
    }

    /// Uses [`rand::thread_rng`] for during initialization.
    #[inline]
    pub fn thread_rng(self) -> NNBuilder<DIM, LP, ThreadRng> {
        self.rng(rand::thread_rng())
    }

    /// Note: currently the same as `.thread_rng()`
    #[inline]
    pub fn default_rng(self) -> NNBuilder<DIM, LP, ThreadRng> {
        self.thread_rng()
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> NNBuilder<DIM, LP, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }

    /// Sets the activation function which is used by the Builder by default when creating new
    /// layers.
    #[inline]
    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

// TODO: maybe allow NoRng + layer_from_parameters
impl NNBuilder<NoDim, NoLayerParts, NoRng> {
    /// Sets the number of inputs the neural network has to `N`.
    ///
    /// This automatically uses `.default_rng()`.
    #[inline]
    pub fn input<const IN: usize>(self) -> BuilderNoParts<IN, ThreadRng> {
        self.default_rng().input()
    }
}

impl<RNG: rand::Rng> NNBuilder<NoDim, NoLayerParts, RNG> {
    /// Sets the number of inputs the neural network has to `N`.
    #[inline]
    pub fn input<const IN: usize>(self) -> BuilderNoParts<IN, RNG> {
        NNBuilder { input_dim: PhantomData, ..self }
    }
}

impl<const IN: usize, LP, RNG> NNBuilder<In<IN>, LP, RNG> {
    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }
}

/// Alias for a [`NNBuilder`] without [`LayerParts`].
pub type BuilderNoParts<const IN: usize, RNG> = NNBuilder<In<IN>, NoLayerParts, RNG>;
/// Alias for a [`NNBuilder`] with [`LayerParts`].
pub type BuilderWithParts<const IN: usize, RNG> = NNBuilder<In<IN>, LayerParts, RNG>;

/// This ensures a consistent interface between [`BuilderNoParts`] and [`BuilderWithParts`].
pub trait BuildLayer<const IN: usize, RNG: rand::Rng>: Sized {
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn _into_without_parts(self) -> BuilderNoParts<IN, RNG>;

    /// Add a [`Layer`] to the neural network.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn _layer(self, layer: Layer) -> BuilderNoParts<IN, RNG> {
        let mut builder = self._into_without_parts();
        assert_eq!(
            layer.get_input_count(),
            builder.last_neuron_count(),
            "input count doesn't match output count of previous layer"
        );
        builder.layers.push(layer);
        builder
    }

    /// Use [`Initializer`] to add a new [`Layer`] to the neural network.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn layer(
        self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderWithParts<IN, RNG> {
        assert_ne!(neurons, 0, "Cannot create a layer with zero neurons.");
        let mut builder = self._into_without_parts();
        let input_count = builder.last_neuron_count();
        let weights = weights_init.init_weights(&mut builder.rng, input_count, neurons);
        let bias = bias_init.init_bias(&mut builder.rng, input_count, neurons);
        let layer_parts = LayerParts { weights, bias };
        NNBuilder { layer_parts, ..builder }
    }

    /// Use the same [`Initializer`] to add multiple new [`Layer`]s to the NeuralNetwork.
    /// Every new layer gets `activation_function`.
    /// This method calls `clone` on `weights_init` and `bias_init`.
    /// See `layer` method.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn layers(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
        activation_function: ActivationFn,
    ) -> BuilderNoParts<IN, RNG> {
        let builder = self._into_without_parts();
        neurons
            .into_iter()
            .fold(builder, |builder: BuilderNoParts<IN, RNG>, neurons: &usize| {
                builder
                    .layer(*neurons, weights_init.clone(), bias_init.clone())
                    .activation_function(activation_function)
            })
    }

    /// similar to `layers` but uses the default activation function for every layer.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn layers_default(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias>,
    ) -> BuilderNoParts<IN, RNG> {
        let builder = self._into_without_parts();
        let default = builder.default_activation_function;
        builder.layers(neurons, weights_init, bias_init, default)
    }

    /// Create a new [`Layer`] from the given `weights` and `bias` and add it to the NeuralNetwork.
    /// See `layer` method.
    fn layer_from_parameters(
        self,
        weights: Matrix<f64>,
        bias: LayerBias,
    ) -> BuilderWithParts<IN, RNG> {
        use Initializer::Initialized;
        self.layer(weights.get_height(), Initialized(weights), Initialized(bias))
    }

    /// Consumes `self` to create a new [`NeuralNetwork`].
    ///
    /// If you want to create a [`NNTrainer`] to train the [`NeuralNetwork`], use `to_trainer`
    /// instead.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `OUT` doesn't match the the neuron count of the last layer.
    fn build<const OUT: usize>(self) -> NeuralNetwork<IN, OUT> {
        let builder = self._into_without_parts();
        assert_eq!(builder.last_neuron_count(), OUT);
        NeuralNetwork::new(builder.layers)
    }

    /// Consumes `self` to create a [`NNTrainerBuilder`].
    ///
    /// Alias for `.build().to_trainer()`
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    ///
    /// # Panics
    ///
    /// See `NeuralNetworkBuilder::build`
    /// [`NeuralNetwork`].
    #[inline]
    fn to_trainer<const OUT: usize>(
        self,
    ) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
        self.build().to_trainer()
    }
}

impl<const IN: usize, RNG: rand::Rng> BuildLayer<IN, RNG> for BuilderNoParts<IN, RNG> {
    #[inline]
    fn _into_without_parts(self) -> BuilderNoParts<IN, RNG> {
        self
    }
}

macro_rules! activation_function {
    ( $( $fn_name:ident -> $variant:ident $( { $($arg:ident : $ty:ty),+ } )? : $variant_str:expr );+ $(;)? ) => {
        $(
            #[doc = "Sets the `"]
            #[doc = $variant_str]
            #[doc = "` activation function for the previously defined layer."]
            #[inline]
            pub fn $fn_name(self $(, $($arg : $ty),+)? ) -> BuilderNoParts<IN, RNG> {
                self.activation_function(ActivationFn::$variant $({ $($arg),+ })?)
            }
         )+
    };
}

impl<const IN: usize, RNG: rand::Rng> BuilderWithParts<IN, RNG> {
    activation_function! {
        identity -> Identity : "Identity" ;
        relu -> ReLU : "ReLU" ;
        leaky_relu -> LeakyReLU { leak_rate: f64 } : "LeakyReLU" ;
        sigmoid -> Sigmoid : "Sigmoid" ;
        softmax -> Softmax : "Softmax" ;
        log_softmax -> LogSoftmax : "LogSoftmax";
    }

    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, af: ActivationFn) -> BuilderNoParts<IN, RNG> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, af);
        NNBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// This function gets called automatically if no activation function is provided.
    #[inline]
    pub fn use_default_activation_function(self) -> BuilderNoParts<IN, RNG> {
        let default = self.default_activation_function;
        self.activation_function(default)
    }
}

impl<const IN: usize, RNG: rand::Rng> BuildLayer<IN, RNG> for BuilderWithParts<IN, RNG> {
    #[inline]
    fn _into_without_parts(self) -> BuilderNoParts<IN, RNG> {
        self.use_default_activation_function()
    }
}
