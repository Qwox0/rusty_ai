use super::trainable::ClipGradientNorm;
use crate::optimizer::IsOptimizer;
use crate::prelude::*;
use crate::util::RngWrapper;
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use std::{iter::once, marker::PhantomData};

// Dimension Markers
pub struct NoDim;
pub struct In<const IN: usize>;
pub struct InOut<const IN: usize, const OUT: usize>;

// Layer parts Markers
pub struct NoParts;
pub struct LayerParts {
    weights: Matrix<f64>,
    bias: Option<LayerBias>,
    activation_function: Option<ActivationFn>,
}

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
pub struct NeuralNetworkBuilder<DIM, LP, OPT, D> {
    layers: Vec<Layer>,
    dim: PhantomData<DIM>,
    layer_parts: LP,
    error_function: Option<ErrorFunction>,

    // for generation
    rng: RngIter<D>,
    default_activation_function: ActivationFn,

    // for trainable neural network
    optimizer: OPT,
    retain_gradient: bool,
    clip_grad_norm: Option<ClipGradientNorm>,
}

impl Default for NeuralNetworkBuilder<NoDim, NoParts, NoOptimizer, Uniform<f64>> {
    fn default() -> Self {
        NeuralNetworkBuilder {
            layers: vec![],
            dim: PhantomData,
            layer_parts: NoParts,
            error_function: None,
            rng: RngIter::New {
                seed: None,
                distr: Uniform::new(0.0, 1.0),
            },
            default_activation_function: ActivationFn::default(),
            optimizer: NoOptimizer,
            retain_gradient: true,
            clip_grad_norm: None,
        }
    }
}

macro_rules! update_phantom {
    ( $builder:expr ) => {
        NeuralNetworkBuilder {
            dim: PhantomData,
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

    pub fn retain_gradient(mut self, retain_gradient: bool) -> Self {
        self.retain_gradient = retain_gradient;
        self
    }
}

impl<D> NeuralNetworkBuilder<NoDim, NoParts, NoOptimizer, D>
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

    pub fn input<const N: usize>(mut self) -> NeuralNetworkBuilder<In<N>, NoParts, NoOptimizer, D> {
        self.rng = self.rng.lock();
        update_phantom!(self)
    }
}

impl<D, const IN: usize> NeuralNetworkBuilder<In<IN>, NoParts, NoOptimizer, D>
where
    D: Distribution<f64>,
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
        let mut rng_iter = self.rng.get_rng_iter();
        let weights = Matrix::from_iter(inputs, neurons, &mut rng_iter);
        let bias = LayerBias::from_iter(neurons, rng_iter);
        let layer = Layer::new(weights, bias, self.default_activation_function);
        self.layer(layer)
    }

    /// # Panics
    /// Panics if `neurons_per_layer.len() == 0`
    pub fn random_layers(mut self, neurons_per_layer: &[usize]) -> Self {
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

    pub fn layer_with_weights_and_bias(self, weights: Matrix<f64>, bias: LayerBias) -> Self {
        self.layer_weights(weights).layer_bias(bias).build_layer()
    }

    fn last_neuron_count(&self) -> usize {
        self.layers
            .last()
            .map(Layer::get_neuron_count)
            .unwrap_or(IN)
    }

    /// # Panics
    /// Panics if OUT doesn't match the the neuron count of the last layer.
    pub fn output<const OUT: usize>(
        self,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, NoOptimizer, D> {
        assert_eq!(self.last_neuron_count(), OUT);
        update_phantom!(self)
    }
}

impl<LP, D, const IN: usize> NeuralNetworkBuilder<In<IN>, LP, NoOptimizer, D> {
    pub fn layer_weights(
        self,
        weights: Matrix<f64>,
    ) -> NeuralNetworkBuilder<In<IN>, LayerParts, NoOptimizer, D> {
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

impl<D, const IN: usize> NeuralNetworkBuilder<In<IN>, LayerParts, NoOptimizer, D>
where
    D: Distribution<f64>,
{
    pub fn layer_bias(mut self, bias: LayerBias) -> Self {
        let _ = self.layer_parts.bias.insert(bias);
        self
    }

    pub fn build_layer(mut self) -> NeuralNetworkBuilder<In<IN>, NoParts, NoOptimizer, D> {
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

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, NoOptimizer, D>
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
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, HasOptimizer, D> {
        let optimizer = HasOptimizer(optimizer);
        NeuralNetworkBuilder { optimizer, ..self }
    }

    pub fn adam_optimizer(
        self,
        optimizer: Adam,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, HasOptimizer, D> {
        self.optimizer(Optimizer::Adam(optimizer))
    }

    pub fn sgd_optimizer(
        self,
        optimizer: GradientDescent,
    ) -> NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, HasOptimizer, D> {
        self.optimizer(Optimizer::GradientDescent(optimizer))
    }
}

impl<D, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<InOut<IN, OUT>, NoParts, HasOptimizer, D>
where
    D: Distribution<f64>,
{
    pub fn clip_gradient_norm(mut self, max_norm: f64, norm_type: Norm) -> Self {
        let _ = self.clip_grad_norm.insert(ClipGradientNorm {
            norm_type,
            max_norm,
        });
        self
    }

    /// builds a trainable neural network
    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT> {
        let mut optimizer = self.optimizer.0;
        optimizer.init_with_layers(&self.layers);
        let network = NeuralNetwork::new(self.layers, self.error_function.unwrap_or_default());
        TrainableNeuralNetwork::new(
            network,
            optimizer,
            self.retain_gradient,
            self.clip_grad_norm,
        )
    }
}
