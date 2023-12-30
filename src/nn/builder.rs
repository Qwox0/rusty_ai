//! # Neural network builder module

#[allow(unused_imports)]
use crate::trainer::NNTrainer;
use crate::{
    bias::LayerBias,
    layer::Layer,
    matrix::Matrix,
    trainer::{
        markers::{NoLossFunction, NoOptimizer},
        NNTrainerBuilder,
    },
    ActivationFn, Initializer, NeuralNetwork,
};
use half::{bf16, f16};
use markers::*;
use matrix::{Element, Float};
use rand::{
    rngs::{StdRng, ThreadRng},
    SeedableRng,
};
use std::marker::PhantomData;

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
    pub struct LayerParts<X> {
        pub(super) weights: Matrix<X>,
        pub(super) bias: LayerBias<X>,
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
pub struct NNBuilder<X, IN, LP, RNG> {
    layers: Vec<Layer<X>>,
    layer_parts: LP,

    input_dim: PhantomData<IN>,

    // for generation
    default_activation_function: ActivationFn<X>,
    rng: RNG,
}

impl Default for NNBuilder<f32, NoDim, NoLayerParts, NoRng> {
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

impl<X, DIM, LP, RNG> NNBuilder<X, DIM, LP, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> NNBuilder<X, DIM, LP, R> {
        NNBuilder { rng, ..self }
    }

    /// Uses [`rand::thread_rng`] for during initialization.
    #[inline]
    pub fn thread_rng(self) -> NNBuilder<X, DIM, LP, ThreadRng> {
        self.rng(rand::thread_rng())
    }

    /// Note: currently the same as `.thread_rng()`
    #[inline]
    pub fn default_rng(self) -> NNBuilder<X, DIM, LP, ThreadRng> {
        self.thread_rng()
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> NNBuilder<X, DIM, LP, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }

    /// Sets the activation function which is used by the Builder by default when creating new
    /// layers.
    #[inline]
    pub fn default_activation_function(mut self, act_func: ActivationFn<X>) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

impl<X, RNG> NNBuilder<X, NoDim, NoLayerParts, RNG> {
    /// Sets `NX` as nn element type.
    pub fn element_type<NX>(self) -> NNBuilder<NX, NoDim, NoLayerParts, RNG> {
        let default_activation_function = Default::default(); // shouldn't be set yet
        NNBuilder { layers: vec![], default_activation_function, ..self }
    }

    /// Sets [`f64`] as nn element type.
    pub fn double_precision(self) -> NNBuilder<f64, NoDim, NoLayerParts, RNG> {
        self.element_type()
    }

    /// Sets [`half::f16`] as nn element type.
    pub fn half_precision(self) -> NNBuilder<f16, NoDim, NoLayerParts, RNG> {
        self.element_type()
    }

    /// Sets [`half::bf16`] as nn element type.
    pub fn bhalf_precision(self) -> NNBuilder<bf16, NoDim, NoLayerParts, RNG> {
        self.element_type()
    }
}

// TODO: maybe allow NoRng + layer_from_parameters
impl<X> NNBuilder<X, NoDim, NoLayerParts, NoRng> {
    /// Sets the number of inputs the neural network has to `N`.
    ///
    /// This automatically uses `.default_rng()`.
    #[inline]
    pub fn input<const IN: usize>(self) -> BuilderNoParts<X, IN, ThreadRng> {
        self.default_rng().input()
    }
}

impl<X, RNG: rand::Rng> NNBuilder<X, NoDim, NoLayerParts, RNG> {
    /// Sets the number of inputs the neural network has to `N`.
    #[inline]
    pub fn input<const IN: usize>(self) -> BuilderNoParts<X, IN, RNG> {
        NNBuilder { input_dim: PhantomData, ..self }
    }
}

impl<X, const IN: usize, LP, RNG> NNBuilder<X, In<IN>, LP, RNG> {
    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }
}

/// Alias for a [`NNBuilder`] without [`LayerParts`].
pub type BuilderNoParts<X, const IN: usize, RNG> = NNBuilder<X, In<IN>, NoLayerParts, RNG>;
/// Alias for a [`NNBuilder`] with [`LayerParts`].
pub type BuilderWithParts<X, const IN: usize, RNG> = NNBuilder<X, In<IN>, LayerParts<X>, RNG>;

/// This ensures a consistent interface between [`BuilderNoParts`] and [`BuilderWithParts`].
pub trait BuildLayer<X, const IN: usize, RNG: rand::Rng>: Sized {
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn _into_without_parts(self) -> BuilderNoParts<X, IN, RNG>;

    /// Add a [`Layer`] to the neural network.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn _layer(self, layer: Layer<X>) -> BuilderNoParts<X, IN, RNG> {
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
        weights_init: Initializer<X, Matrix<X>>,
        bias_init: Initializer<X, LayerBias<X>>,
    ) -> BuilderWithParts<X, IN, RNG>
    where
        X: Float,
        rand_distr::StandardNormal: rand_distr::Distribution<X>,
    {
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
        weights_init: Initializer<X, Matrix<X>>,
        bias_init: Initializer<X, LayerBias<X>>,
        activation_function: ActivationFn<X>,
    ) -> BuilderNoParts<X, IN, RNG>
    where
        X: Float,
        rand_distr::StandardNormal: rand_distr::Distribution<X>,
    {
        let builder = self._into_without_parts();
        neurons
            .into_iter()
            .fold(builder, |builder: BuilderNoParts<X, IN, RNG>, neurons: &usize| {
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
        weights_init: Initializer<X, Matrix<X>>,
        bias_init: Initializer<X, LayerBias<X>>,
    ) -> BuilderNoParts<X, IN, RNG>
    where
        X: Float,
        rand_distr::StandardNormal: rand_distr::Distribution<X>,
    {
        let builder = self._into_without_parts();
        let default = builder.default_activation_function;
        builder.layers(neurons, weights_init, bias_init, default)
    }

    /// Create a new [`Layer`] from the given `weights` and `bias` and add it to the NeuralNetwork.
    /// See `layer` method.
    fn layer_from_parameters(
        self,
        weights: Matrix<X>,
        bias: LayerBias<X>,
    ) -> BuilderWithParts<X, IN, RNG>
    where
        X: Float,
        rand_distr::StandardNormal: rand_distr::Distribution<X>,
    {
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
    fn build<const OUT: usize>(self) -> NeuralNetwork<X, IN, OUT> {
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
    ) -> NNTrainerBuilder<X, IN, OUT, NoLossFunction, NoOptimizer> {
        self.build().to_trainer()
    }
}

impl<X, const IN: usize, RNG: rand::Rng> BuildLayer<X, IN, RNG> for BuilderNoParts<X, IN, RNG> {
    #[inline]
    fn _into_without_parts(self) -> BuilderNoParts<X, IN, RNG> {
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
            pub fn $fn_name(self $(, $($arg : $ty),+)? ) -> BuilderNoParts<X, IN, RNG> {
                self.activation_function(ActivationFn::$variant $({ $($arg),+ })?)
            }
         )+
    };
}

impl<X: Element, const IN: usize, RNG: rand::Rng> BuilderWithParts<X, IN, RNG> {
    activation_function! {
        identity -> Identity : "Identity" ;
        relu -> ReLU : "ReLU" ;
        leaky_relu -> LeakyReLU { leak_rate: X } : "LeakyReLU" ;
        sigmoid -> Sigmoid : "Sigmoid" ;
        softmax -> Softmax : "Softmax" ;
        log_softmax -> LogSoftmax : "LogSoftmax";
    }

    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, af: ActivationFn<X>) -> BuilderNoParts<X, IN, RNG> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, af);
        NNBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// This function gets called automatically if no activation function is provided.
    #[inline]
    pub fn use_default_activation_function(self) -> BuilderNoParts<X, IN, RNG> {
        let default = self.default_activation_function;
        self.activation_function(default)
    }
}

impl<X: Element, const IN: usize, RNG: rand::Rng> BuildLayer<X, IN, RNG> for BuilderWithParts<X, IN, RNG> {
    #[inline]
    fn _into_without_parts(self) -> BuilderNoParts<X, IN, RNG> {
        self.use_default_activation_function()
    }
}

/*
//! # Neural network builder module

#[allow(unused_imports)]
use crate::trainer::NNTrainer;
use crate::{
    bias::LayerBias<X>,
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
    /// Marker for an undefined Dimension.
    pub struct NoDim;
    /// Inputs dimension == `IN`
    pub struct In<const IN: usize>;

    /// Rng hasn't been set yet
    pub struct NoRng;
}

#[derive(Debug)]
enum LastLayer {
    Finished,
    Unfinished { weights: Matrix<f64>, bias: LayerBias<X> },
}

/// Builder
///
/// # Generics
///
/// `IN`: input dimension (not set yet or const usize)
/// `LP`: layer parts (contains weights and bias til next activation function is set)
/// `RNG`: rng type (seeded or not)
#[derive(Debug)]
pub struct NNBuilder<IN, RNG> {
    layers: Vec<Layer>,
    last_layer: LastLayer,

    input_dim: PhantomData<IN>,

    // for generation
    default_activation_function: ActivationFn<X>,
    rng: RNG,
}

impl Default for NNBuilder<NoDim, NoRng> {
    fn default() -> Self {
        NNBuilder {
            layers: vec![],
            last_layer: LastLayer::Finished,
            input_dim: PhantomData,
            default_activation_function: ActivationFn<X>::default(),
            rng: NoRng,
        }
    }
}

impl<X, DIM, RNG> NNBuilder<DIM, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<RNG2: rand::Rng>(self, rng: RNG2) -> NNBuilder<DIM, RNG2> {
        NNBuilder { rng, ..self }
    }

    /// Uses [`rand::thread_rng`] for during initialization.
    #[inline]
    pub fn thread_rng(self) -> NNBuilder<DIM, ThreadRng> {
        self.rng(rand::thread_rng())
    }

    /// Note: currently the same as `.thread_rng()`
    #[inline]
    pub fn default_rng(self) -> NNBuilder<DIM, ThreadRng> {
        self.thread_rng()
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> NNBuilder<DIM, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }

    /// Sets the activation function which is used by the Builder by default when creating new
    /// layers.
    #[inline]
    pub fn default_activation_function(mut self, act_func: ActivationFn<X>) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

// TODO: maybe allow NoRng + layer_from_parameters
impl NNBuilder<NoDim, NoRng> {
    /// Sets the number of inputs the neural network has to `N`.
    ///
    /// This automatically uses `.default_rng()`.
    #[inline]
    pub fn input<const IN: usize>(self) -> NNBuilder<In<IN>, ThreadRng> {
        self.default_rng().input()
    }
}

impl<X, RNG: rand::Rng> NNBuilder<NoDim, RNG> {
    /// Sets the number of inputs the neural network has to `N`.
    #[inline]
    pub fn input<const IN: usize>(self) -> NNBuilder<In<IN>, RNG> {
        NNBuilder { input_dim: PhantomData, ..self }
    }
}

impl<X, const IN: usize, RNG> NNBuilder<In<IN>, RNG> {
    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }
}

impl<X, const IN: usize, RNG: rand::Rng> NNBuilder<In<IN>, RNG> {
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    fn finish_last_layer(self) -> Self {}

    /// Add a [`Layer`] to the neural network.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    pub fn _layer(self, layer: Layer) -> Self {
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
    pub fn layer(
        self,
        neurons: usize,
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias<X>>,
    ) -> Self {
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
    pub fn layers(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias<X>>,
        activation_function: ActivationFn<X>,
    ) -> Self {
        let builder = self._into_without_parts();
        neurons.into_iter().fold(builder, |builder: Self, neurons: &usize| {
            builder
                .layer(*neurons, weights_init.clone(), bias_init.clone())
                .activation_function(activation_function)
        })
    }

    /// similar to `layers` but uses the default activation function for every layer.
    ///
    /// This uses the `self.default_activation_function` for the previously defined layer if not
    /// explicitly set otherwise.
    pub fn layers_default(
        self,
        neurons: &[usize],
        weights_init: Initializer<Matrix<f64>>,
        bias_init: Initializer<LayerBias<X>>,
    ) -> Self {
        let builder = self._into_without_parts();
        let default = builder.default_activation_function;
        builder.layers(neurons, weights_init, bias_init, default)
    }

    /// Create a new [`Layer`] from the given `weights` and `bias` and add it to the NeuralNetwork.
    /// See `layer` method.
    pub fn layer_from_parameters(self, weights: Matrix<f64>, bias: LayerBias<X>) -> Self {
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
    pub fn build<const OUT: usize>(self) -> NeuralNetwork<X, IN, OUT> {
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
    pub fn to_trainer<const OUT: usize>(
        self,
    ) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
        self.build().to_trainer()
    }
}

macro_rules! activation_function {
    ( $( $fn_name:ident -> $variant:ident $( { $($arg:ident : $ty:ty),+ } )? : $variant_str:expr );+ $(;)? ) => {
        $(
            #[doc = "Sets the `"]
            #[doc = $variant_str]
            #[doc = "` activation function for the previously defined layer."]
            #[inline]
            pub fn $fn_name(self $(, $($arg : $ty),+)? ) -> NNBuilder<In<IN>, RNG> {
                self.activation_function(ActivationFn<X>::$variant $({ $($arg),+ })?)
            }
         )+
    };
}

impl<X, const IN: usize, RNG: rand::Rng> NNBuilder<In<IN>, RNG> {
    activation_function! {
        identity -> Identity : "Identity" ;
        relu -> ReLU : "ReLU" ;
        leaky_relu -> LeakyReLU { leak_rate: f64 } : "LeakyReLU" ;
        sigmoid -> Sigmoid : "Sigmoid" ;
        softmax -> Softmax : "Softmax" ;
        log_softmax -> LogSoftmax : "LogSoftmax";
    }

    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, af: ActivationFn<X>) -> NNBuilder<In<IN>, RNG> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, af);
        NNBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// This function gets called automatically if no activation function is provided.
    #[inline]
    pub fn use_default_activation_function(self) -> NNBuilder<In<IN>, RNG> {
        let default = self.default_activation_function;
        self.activation_function(default)
    }
}
 */
