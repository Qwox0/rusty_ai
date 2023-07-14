use crate::{
    clip_gradient_norm::ClipGradientNorm, initializer::Initializer, optimizer::IsOptimizer,
    prelude::*, util::RngWrapper,
};
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution};
use std::{iter::once, marker::PhantomData};

// Dimension Markers
pub struct NoDim;
pub struct In<const IN: usize>;
pub struct InOut<const IN: usize, const OUT: usize>;

// Layer parts Markers
/*
pub struct NoParts;
pub struct LayerParts {
    weights: Matrix<f64>,
    bias: Option<LayerBias>,
    activation_function: Option<ActivationFn>,
}
*/

// Optimizer Markers
pub struct NoOptimizer;
pub struct HasOptimizer(Optimizer);

// Rng Markers
pub struct Seed(Option<u64>);

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<DIM, RNG, INIT, OPT>
where INIT: Initializer
{
    layers: Vec<Layer>,
    dim: PhantomData<DIM>,
    error_function: Option<ErrorFunction>,

    // for generation
    rng: RNG,
    default_initializer: INIT,
    default_activation_function: ActivationFn,

    // for trainable neural network
    optimizer: OPT,
    retain_gradient: bool,
    clip_grad_norm: Option<ClipGradientNorm>,
}

impl Default for NeuralNetworkBuilder<NoDim, Seed, Rand<Uniform<f64>>, NoOptimizer> {
    fn default() -> Self {
        NeuralNetworkBuilder {
            layers: vec![],
            dim: PhantomData,
            error_function: None,
            rng: Seed(None),
            default_initializer: Rand(Uniform::from(0.0..1.0)),
            default_activation_function: ActivationFn::default(),
            optimizer: NoOptimizer,
            retain_gradient: true,
            clip_grad_norm: None,
        }
    }
}

macro_rules! update_phantom {
    ($builder:expr) => {
        NeuralNetworkBuilder { dim: PhantomData, ..$builder }
    };
}

impl<DIM, RNG, INIT, OPT> NeuralNetworkBuilder<DIM, RNG, INIT, OPT>
where INIT: Initializer
{
    pub fn error_function(mut self, error_function: ErrorFunction) -> Self {
        let _ = self.error_function.insert(error_function);
        self
    }

    pub fn default_initializer<NINIT: Initializer>(
        self,
        initializer: NINIT,
    ) -> NeuralNetworkBuilder<DIM, RNG, NINIT, OPT> {
        NeuralNetworkBuilder { default_initializer: initializer, ..self }
    }

    pub fn default_activation_function(mut self, act_func: ActivationFn) -> Self {
        self.default_activation_function = act_func;
        self
    }

    pub fn retain_gradient(mut self, retain_gradient: bool) -> Self {
        self.retain_gradient = retain_gradient;
        self
    }
}

impl<INIT> NeuralNetworkBuilder<NoDim, Seed, INIT, NoOptimizer>
where INIT: Initializer
{
    pub fn rng_seed(mut self, seed: u64) -> Self {
        let _ = self.rng.0.insert(seed);
        self
    }

    pub fn input<const N: usize>(
        self,
    ) -> NeuralNetworkBuilder<In<N>, RngWrapper, INIT, NoOptimizer> {
        let rng = RngWrapper::new(self.rng.0);
        NeuralNetworkBuilder { dim: PhantomData, rng, ..self }
    }
}

impl<INIT, const IN: usize> NeuralNetworkBuilder<In<IN>, RngWrapper, INIT, NoOptimizer>
where INIT: Initializer
{
    pub fn layer(mut self, layer: Layer) -> Self {
        assert_eq!(
            layer.get_input_count(),
            self.last_neuron_count(),
            "input count doesn't match previously set value"
        );
        self.layers.push(layer);
        self
    }

    pub fn random_layer(mut self, neurons: usize) -> Self {
        let inputs = self.last_neuron_count();
        let act_fn = self.default_activation_function;
        let layer = self.default_initializer.init_layer(&mut self.rng, inputs, neurons, act_fn);
        self.layer(layer)
    }

    /// # Panics
    /// Panics if `neurons_per_layer.len() == 0`
    pub fn random_layers(mut self, neurons_per_layer: &[usize]) -> Self {
        assert!(neurons_per_layer.len() > 0);
        let last_count = self.last_neuron_count();
        let act_fn = self.default_activation_function;
        for (&last, &count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            let layer = self.default_initializer.init_layer(&mut self.rng, last, count, act_fn);
            self.layers.push(layer);
        }
        update_phantom!(self)
    }

    pub fn custom_layer(self, weights: Matrix<f64>, bias: LayerBias) -> Self {
        let layer = Layer::new(weights, bias, self.default_activation_function);
        self.layer(layer)
    }

    fn last_neuron_count(&self) -> usize {
        self.layers.last().map(Layer::get_neuron_count).unwrap_or(IN)
    }

    /// # Panics
    /// Panics if `OUT` doesn't match the the neuron count of the last layer.
    pub fn output<const OUT: usize>(
        self,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, NoOptimizer> {
        assert_eq!(self.last_neuron_count(), OUT);
        update_phantom!(self)
    }
}

/*
impl<LP, D, const IN: usize> NeuralNetworkBuilder<In<IN>, RngWrapper, D, NoOptimizer>
where
    D: Distribution<f64>,
{
    pub fn layer_weights(
        self,
        weights: Matrix<f64>,
    ) -> NeuralNetworkBuilder<In<IN>, RngWrapper, D, NoOptimizer> {
        let layer_parts = LayerParts {
            weights,
            bias: None,
            activation_function: None,
        };
        NeuralNetworkBuilder {
            layer_parts,
            ..self
        }
    }
}

impl<D, const IN: usize> NeuralNetworkBuilder<In<IN>, RngWrapper, D, NoOptimizer>
where
    D: Distribution<f64>,
{
    pub fn layer_bias(mut self, bias: LayerBias) -> Self {
        let _ = self.layer_parts.bias.insert(bias);
        self
    }

    pub fn build_layer(
        mut self,
    ) -> NeuralNetworkBuilder<In<IN>, RngWrapper, D, NoOptimizer> {
        let bias = self.layer_parts.bias.take().unwrap_or_else(|| {
            let neurons = self.layer_parts.weights.get_width();
            LayerBias::from_iter(neurons, self.rng.get_rng_iter())
        });
        let weights = self.layer_parts.weights;
        let activation_function = self
            .layer_parts
            .activation_function
            .take()
            .unwrap_or(self.default_activation_function);
        let layer = Layer::new(weights, bias, activation_function);
        NeuralNetworkBuilder {
            layer_parts: NoParts,
            ..self
        }
        .layer(layer)
    }
}
*/

impl<INIT, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, NoOptimizer>
where INIT: Initializer
{
    /// builds a non-trainable neural network
    pub fn build(self) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default())
    }

    pub fn optimizer(
        self,
        optimizer: Optimizer,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, HasOptimizer> {
        let optimizer = HasOptimizer(optimizer);
        NeuralNetworkBuilder { optimizer, ..self }
    }

    pub fn adam_optimizer(
        self,
        optimizer: Adam,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, HasOptimizer> {
        self.optimizer(Optimizer::Adam(optimizer))
    }

    pub fn sgd_optimizer(
        self,
        optimizer: GradientDescent,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, HasOptimizer> {
        self.optimizer(Optimizer::GradientDescent(optimizer))
    }
}

impl<INIT, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<InOut<IN, OUT>, RngWrapper, INIT, HasOptimizer>
where INIT: Initializer
{
    pub fn clip_gradient_norm(mut self, max_norm: f64, norm_type: Norm) -> Self {
        let clip_grad_norm = ClipGradientNorm::new(norm_type, max_norm);
        let _ = self.clip_grad_norm.insert(clip_grad_norm);
        self
    }

    /// builds a trainable neural network
    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT> {
        let mut optimizer = self.optimizer.0;
        optimizer.init_with_layers(&self.layers);
        let network = NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default());
        TrainableNeuralNetwork::new(network, optimizer, self.retain_gradient, self.clip_grad_norm)
    }
}
