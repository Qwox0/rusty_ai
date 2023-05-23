use crate::layer::LayerOrLayerBuilder;
use crate::prelude::*;
use crate::{
    layer::{InputLayer, IsLayer},
    optimizer::IsOptimizer,
};
use itertools::Itertools;
use std::{iter::once, marker::PhantomData};

// Layer Markers
pub struct NoLayer;
pub struct Input<const N: usize>;
pub struct Hidden;
pub struct Output<const N: usize>;

pub trait InputOrHidden {}
impl<const N: usize> InputOrHidden for Input<N> {}
impl InputOrHidden for Hidden {}

// Error Function Markers
pub struct NoErrFn;
pub struct HasErrFn(ErrorFunction);
pub trait GetErrFn: Sized {
    fn get(self) -> ErrorFunction;
}
impl GetErrFn for NoErrFn {
    fn get(self) -> ErrorFunction {
        ErrorFunction::default()
    }
}
impl GetErrFn for HasErrFn {
    fn get(self) -> ErrorFunction {
        self.0
    }
}

// Optimizer Markers
pub struct NoOptimizer;
pub struct HasOptimizer(Optimizer);

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST, ERRORFN, OPT> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
    error_function: ERRORFN,
    optimizer: OPT,
}

impl NeuralNetworkBuilder<NoLayer, NoLayer, NoErrFn, NoOptimizer> {
    pub fn new() -> NeuralNetworkBuilder<NoLayer, NoLayer, NoErrFn, NoOptimizer> {
        NeuralNetworkBuilder {
            layers: vec![],
            input: PhantomData,
            last_layer: PhantomData,
            error_function: NoErrFn,
            optimizer: NoOptimizer,
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

impl<EF> NeuralNetworkBuilder<NoLayer, NoLayer, EF, NoOptimizer> {
    pub fn input_layer<const N: usize>(
        self,
    ) -> NeuralNetworkBuilder<Input<N>, Input<N>, EF, NoOptimizer> {
        update_phantom!(self)
    }
}

impl<LL: InputOrHidden, EF, const IN: usize> NeuralNetworkBuilder<Input<IN>, LL, EF, NoOptimizer> {
    pub fn hidden_layer(
        mut self,
        layer: impl LayerOrLayerBuilder,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, EF, NoOptimizer> {
        let inputs = self.last_neuron_count();
        let layer = layer.as_layer_with_inputs(inputs);
        self.layers.push(layer);
        update_phantom!(self)
    }

    pub fn hidden_layer_random(
        self,
        neurons: usize,
        act_fn: ActivationFn,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, EF, NoOptimizer> {
        let layer = Layer::random(self.last_neuron_count(), neurons, act_fn);
        self.hidden_layer(layer)
    }

    /// panics if `neurons_per_layer.len() == 0`
    pub fn hidden_layers_random(
        mut self,
        neurons_per_layer: &[usize],
        act_fn: ActivationFn,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, EF, NoOptimizer> {
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
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, NoOptimizer> {
        let inputs = self.last_neuron_count();
        let layer = layer.as_layer_with_inputs(inputs);
        assert_eq!(layer.get_neuron_count(), OUT);
        self.layers.push(layer);
        update_phantom!(self)
    }

    pub fn output_layer_random<const OUT: usize>(
        self,
        act_func: ActivationFn,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, NoOptimizer> {
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

impl<EF: GetErrFn, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, NoOptimizer>
{
    /// builds a non-trainable neural network
    pub fn build(self) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork::new(InputLayer::<IN>, self.layers, self.error_function.get())
    }

    pub fn optimizer(
        self,
        optimizer: Optimizer,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer> {
        let optimizer = HasOptimizer(optimizer);
        NeuralNetworkBuilder { optimizer, ..self }
    }

    pub fn adam_optimizer(
        self,
        optimizer: Adam,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer> {
        self.optimizer(Optimizer::Adam(optimizer))
    }

    pub fn gradient_descent_optimizer(
        self,
        optimizer: GradientDescent,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer> {
        self.optimizer(Optimizer::GradientDescent(optimizer))
    }
}

impl<EF: GetErrFn, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer>
{
    /// builds a trainable neural network
    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT> {
        let mut optimizer = self.optimizer.0;
        optimizer.init_with_layers(&self.layers);
        let network = NeuralNetwork::new(InputLayer::<IN>, self.layers, self.error_function.get());
        TrainableNeuralNetwork::new(network, optimizer)
    }
}

impl<IL, LL, OPT> NeuralNetworkBuilder<IL, LL, NoErrFn, OPT> {
    pub fn error_function(
        self,
        error_function: ErrorFunction,
    ) -> NeuralNetworkBuilder<IL, LL, HasErrFn, OPT> {
        NeuralNetworkBuilder {
            error_function: HasErrFn(error_function),
            ..self
        }
    }
}
