use crate::layer::{BiasType, LayerOrLayerBuilder};
use crate::prelude::*;
use crate::util::RngWrapper;
use crate::{
    layer::{InputLayer, IsLayer},
    optimizer::IsOptimizer,
};
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use std::{iter::once, marker::PhantomData};

// Layer Markers
pub struct NoLayer;
pub struct Input<const N: usize>;
pub struct Hidden;
pub struct Output<const N: usize>;

pub trait InputOrHidden {}
impl<const N: usize> InputOrHidden for Input<N> {}
impl InputOrHidden for Hidden {}

// Optimizer Markers
pub struct NoOptimizer;
pub struct HasOptimizer(Optimizer);

// Rng Markers
pub struct InitRng {
    pub seed: Option<u64>,
}

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST, OPT, RNG, D> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
    error_function: Option<ErrorFunction>,
    optimizer: OPT,

    // for generation
    rng: RNG,
    distr: D,
    default_activation_function: ActivationFn,
}

impl Default for NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer, InitRng, Uniform<f64>> {
    fn default() -> Self {
        NeuralNetworkBuilder {
            layers: vec![],
            input: PhantomData,
            last_layer: PhantomData,
            error_function: None,
            optimizer: NoOptimizer,
            rng: InitRng { seed: None },
            distr: Uniform::from(0.0..1.0),
            default_activation_function: ActivationFn::default(),
        }
    }
}

macro_rules! update_phantom {
    ( $builder:expr ) => {
        NeuralNetworkBuilder {
            input: PhantomData,
            last_layer: PhantomData,
            ..$builder
        }
    };
}

impl<IN, LAST, OPT, INIT, D> NeuralNetworkBuilder<IN, LAST, OPT, INIT, D> {
    pub fn error_function(mut self, error_function: ErrorFunction) -> Self {
        let _ = self.error_function.insert(error_function);
        self
    }

    /// can be changed after first layer!
    pub fn rng_distribution<ND: Distribution<f64>>(
        self,
        distr: ND,
    ) -> NeuralNetworkBuilder<IN, LAST, OPT, INIT, ND> {
        NeuralNetworkBuilder { distr, ..self }
    }
}

impl<D> NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer, InitRng, D>
where
    D: Distribution<f64>,
{
    pub fn rng_seed(mut self, seed: u64) -> Self {
        let _ = self.rng.seed.insert(seed);
        self
    }

    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }

    pub fn input_layer<const N: usize>(
        self,
    ) -> NeuralNetworkBuilder<Input<N>, Input<N>, NoOptimizer, RngWrapper, D> {
        // lock initilizer
        let rng = RngWrapper::new(self.rng.seed);
        NeuralNetworkBuilder {
            input: PhantomData,
            last_layer: PhantomData,
            rng,
            ..self
        }
    }
}

impl<LL, D, const IN: usize> NeuralNetworkBuilder<Input<IN>, LL, NoOptimizer, RngWrapper, D>
where
    LL: InputOrHidden,
    D: Distribution<f64>,
{
    fn get_rng_iter<'a>(&'a mut self) -> impl Iterator<Item = f64> + 'a {
        (&self.distr).sample_iter(&mut self.rng)
    }

    pub fn hidden_layer(
        mut self,
        layer: impl LayerOrLayerBuilder,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, RngWrapper, D> {
        let inputs = self.last_neuron_count();
        let layer = layer.as_layer_with_inputs(inputs);
        self.layers.push(layer);
        update_phantom!(self)
    }

    pub fn hidden_layer_random(
        mut self,
        neurons: usize,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, RngWrapper, D> {
        let inputs = self.last_neuron_count();
        //let rng_iter = (&self.distr).sample_iter(&mut self.rng);
        let act_fn = self.default_activation_function;
        let rng_iter = self.get_rng_iter();
        let layer = Layer::from_iter(inputs, neurons, rng_iter, act_fn);
        self.hidden_layer(layer)
    }

    /// panics if `neurons_per_layer.len() == 0`
    pub fn hidden_layers_random(
        mut self,
        neurons_per_layer: &[usize],
        act_fn: ActivationFn,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, RngWrapper, D> {
        assert!(neurons_per_layer.len() > 0);
        let last_count = self.last_neuron_count();
        for (last, count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers.push(Layer::random(*last, *count, act_fn));
        }
        update_phantom!(self)
    }

    /// make sure OUT matches layer
    pub fn output_layer<const OUT: usize>(
        mut self,
        layer: impl LayerOrLayerBuilder,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer, RngWrapper, D> {
        let inputs = self.last_neuron_count();
        let layer = layer.as_layer_with_inputs(inputs);
        assert_eq!(layer.get_neuron_count(), OUT);
        self.layers.push(layer);
        update_phantom!(self)
    }

    pub fn output_layer_random<const OUT: usize>(
        self,
        act_func: ActivationFn,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer, RngWrapper, D> {
        let layer = Layer::random(self.last_neuron_count(), OUT, act_func);
        self.output_layer(layer)
    }

    pub fn last_neuron_count(&self) -> usize {
        self.layers
            .last()
            .map(Layer::get_neuron_count)
            .unwrap_or(IN)
    }
}

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer, RngWrapper, D>
where
    D: Distribution<f64>,
{
    /// builds a non-trainable neural network
    pub fn build(self) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork::new(
            InputLayer::<IN>,
            self.layers,
            self.error_function.unwrap_or_default(),
        )
    }

    pub fn optimizer(
        self,
        optimizer: Optimizer,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, RngWrapper, D> {
        let optimizer = HasOptimizer(optimizer);
        NeuralNetworkBuilder { optimizer, ..self }
    }

    pub fn adam_optimizer(
        self,
        optimizer: Adam,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, RngWrapper, D> {
        self.optimizer(Optimizer::Adam(optimizer))
    }

    pub fn sgd_optimizer(
        self,
        optimizer: GradientDescent,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, RngWrapper, D> {
        self.optimizer(Optimizer::GradientDescent(optimizer))
    }
}

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, RngWrapper, D>
where
    D: Distribution<f64>,
{
    /// builds a trainable neural network
    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT> {
        let mut optimizer = self.optimizer.0;
        optimizer.init_with_layers(&self.layers);
        let network = NeuralNetwork::new(
            InputLayer::<IN>,
            self.layers,
            self.error_function.unwrap_or_default(),
        );
        TrainableNeuralNetwork::new(network, optimizer)
    }
}
