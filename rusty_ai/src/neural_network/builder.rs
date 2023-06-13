use crate::layer::LayerOrLayerBuilder;
use crate::prelude::*;
use crate::util::RngWrapper;
use crate::{
    layer::IsLayer,
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
#[derive(Debug)]
enum RngIter<D> {
    New { seed: Option<u64>, distr: D },
    Locked { rng: RngWrapper, distr: D },
}

impl<D: Distribution<f64>> RngIter<D> {
    fn set_distr<ND: Distribution<f64>>(self, distr: ND) -> RngIter<ND> {
        use RngIter::*;
        match self {
            New { seed, .. } => New { seed, distr },
            Locked { rng, .. } => Locked { rng, distr },
        }
    }

    fn get_seed_mut(&mut self) -> Option<&mut Option<u64>> {
        match self {
            RngIter::New { seed, .. } => Some(seed),
            RngIter::Locked { .. } => None,
        }
    }

    fn lock(self) -> Self {
        match self {
            RngIter::New { seed, distr } => RngIter::Locked {
                rng: RngWrapper::new(seed),
                distr,
            },
            x => x,
        }
    }

    fn get_rng_iter<'a>(&'a mut self) -> impl Iterator<Item = f64> + 'a {
        match self {
            RngIter::New { .. } => panic!("RngIter must be locked first!"),
            RngIter::Locked { rng, ref distr } => distr.sample_iter(rng),
        }
    }
}

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST, OPT, D> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
    error_function: Option<ErrorFunction>,
    optimizer: OPT,

    // for generation
    rng: RngIter<D>,
    default_activation_function: ActivationFn,
}

impl Default for NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer, Uniform<f64>> {
    fn default() -> Self {
        NeuralNetworkBuilder {
            layers: vec![],
            input: PhantomData,
            last_layer: PhantomData,
            error_function: None,
            optimizer: NoOptimizer,
            rng: RngIter::New {
                seed: None,
                distr: Uniform::new(0.0, 1.0),
            },
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

impl<IN, LAST, OPT, D> NeuralNetworkBuilder<IN, LAST, OPT, D>
where
    D: Distribution<f64>,
{
    pub fn error_function(mut self, error_function: ErrorFunction) -> Self {
        let _ = self.error_function.insert(error_function);
        self
    }

    /// can be changed after first layer!
    pub fn rng_distribution<ND: Distribution<f64>>(
        self,
        distr: ND,
    ) -> NeuralNetworkBuilder<IN, LAST, OPT, ND> {
        let rng = self.rng.set_distr(distr);
        NeuralNetworkBuilder { rng, ..self }
    }

    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }
}

impl<D> NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer, D>
where
    D: Distribution<f64>,
{
    pub fn rng_seed(mut self, seed: u64) -> Self {
        let _ = self
            .rng
            .get_seed_mut()
            .expect("RngIter is unlocked")
            .insert(seed);
        self
    }

    pub fn input<const N: usize>(
        mut self,
    ) -> NeuralNetworkBuilder<Input<N>, Input<N>, NoOptimizer, D> {
        self.rng = self.rng.lock();
        update_phantom!(self)
    }
}

impl<LL, D, const IN: usize> NeuralNetworkBuilder<Input<IN>, LL, NoOptimizer, D>
where
    LL: InputOrHidden,
    D: Distribution<f64>,
{
    pub fn layer(
        mut self,
        layer: impl LayerOrLayerBuilder,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, D> {
        let inputs = self.last_neuron_count();
        let layer = layer.as_layer_with_inputs(inputs);
        self.layers.push(layer);
        update_phantom!(self)
    }

    pub fn layer_random(
        mut self,
        neurons: usize,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, D> {
        let inputs = self.last_neuron_count();
        let act_fn = self.default_activation_function;
        let rng_iter = self.rng.get_rng_iter();
        let layer = Layer::from_iter(inputs, neurons, rng_iter, act_fn);
        self.layer(layer)
    }

    /// # Panics
    /// Panics if `neurons_per_layer.len() == 0`
    pub fn layers_random(
        mut self,
        neurons_per_layer: &[usize],
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, NoOptimizer, D> {
        assert!(neurons_per_layer.len() > 0);
        let last_count = self.last_neuron_count();
        let act_fn = self.default_activation_function;
        let mut rng_iter = self.rng.get_rng_iter();
        for (&last, &count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers
                .push(Layer::from_iter(last, count, &mut rng_iter, act_fn));
        }
        drop(rng_iter);
        update_phantom!(self)
    }

    fn last_neuron_count(&self) -> usize {
        self.layers
            .last()
            .map(Layer::get_neuron_count)
            .unwrap_or(IN)
    }

    pub fn output<const OUT: usize>(
        self,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer, D> {
        update_phantom!(self)
    }
}

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer, D>
where
    D: Distribution<f64>,
{
    /// builds a non-trainable neural network
    pub fn build(self) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default())
    }

    pub fn optimizer(
        self,
        optimizer: Optimizer,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, D> {
        let optimizer = HasOptimizer(optimizer);
        NeuralNetworkBuilder { optimizer, ..self }
    }

    pub fn adam_optimizer(
        self,
        optimizer: Adam,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, D> {
        self.optimizer(Optimizer::Adam(optimizer))
    }

    pub fn sgd_optimizer(
        self,
        optimizer: GradientDescent,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, D> {
        self.optimizer(Optimizer::GradientDescent(optimizer))
    }
}

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer, D>
where
    D: Distribution<f64>,
{
    /// builds a trainable neural network
    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT> {
        let mut optimizer = self.optimizer.0;
        optimizer.init_with_layers(&self.layers);
        let network = NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default());
        TrainableNeuralNetwork::new(network, optimizer)
    }
}
