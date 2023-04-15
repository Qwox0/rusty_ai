use std::{iter::once, marker::PhantomData};

use itertools::Itertools;

use crate::{
    activation_function::ActivationFunction,
    error_function::ErrorFunction,
    layer::{InputLayer, Layer},
    neural_network::NeuralNetwork,
    optimizer::{Optimizer, OptimizerDispatch},
};

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
    fn get(self) -> ErrorFunction {
        ErrorFunction::default()
    }
}
impl GetErrFn for NoErrFn {}
impl GetErrFn for HasErrFn {
    fn get(self) -> ErrorFunction {
        self.0
    }
}

// Optimizer Markers
pub struct NoOptimizer;
pub struct HasOptimizer(OptimizerDispatch);

// Builder
#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST, ERRORFN, OPTIMIZER> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
    error_function: ERRORFN,
    optimizer: OPTIMIZER,
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
impl<EF, O> NeuralNetworkBuilder<NoLayer, NoLayer, EF, O> {
    pub fn input_layer<const N: usize>(self) -> NeuralNetworkBuilder<Input<N>, Input<N>, EF, O> {
        NeuralNetworkBuilder {
            input: PhantomData,
            last_layer: PhantomData,
            ..self
        }
    }
}

impl<LL: InputOrHidden, EF, O, const IN: usize> NeuralNetworkBuilder<Input<IN>, LL, EF, O> {
    pub fn hidden_layer(
        mut self,
        neurons: usize,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, EF, O> {
        self.layers
            .push(Layer::new_hidden(self.last_count(), neurons, act_func));
        NeuralNetworkBuilder {
            last_layer: PhantomData,
            ..self
        }
    }

    pub fn hidden_layers(
        mut self,
        neurons_per_layer: &[usize],
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<IN>, Hidden, EF, O> {
        let last_count = self.last_count();
        for (last, count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers.push(Layer::new_hidden(*last, *count, act_func));
        }
        NeuralNetworkBuilder {
            last_layer: PhantomData,
            ..self
        }
    }

    pub fn output_layer<const OUT: usize>(
        mut self,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, O> {
        self.layers
            .push(Layer::new_output(self.last_count(), OUT, act_func));
        NeuralNetworkBuilder {
            last_layer: PhantomData,
            ..self
        }
    }

    fn last_count(&self) -> usize {
        self.layers
            .last()
            .map(Layer::get_neuron_count)
            .unwrap_or(IN)
    }
}

impl<IL, LL, O> NeuralNetworkBuilder<IL, LL, NoErrFn, O> {
    pub fn error_function(
        self,
        error_function: ErrorFunction,
    ) -> NeuralNetworkBuilder<IL, LL, HasErrFn, O> {
        NeuralNetworkBuilder {
            error_function: HasErrFn(error_function),
            ..self
        }
    }
}

impl<EF, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, NoOptimizer>
{
    pub fn optimizer(
        self,
        optimizer: OptimizerDispatch,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer> {
        NeuralNetworkBuilder {
            optimizer: HasOptimizer(optimizer),
            ..self
        }
    }
}

impl<EF: GetErrFn, const IN: usize, const OUT: usize>
    NeuralNetworkBuilder<Input<IN>, Output<OUT>, EF, HasOptimizer>
{
    pub fn build(mut self) -> NeuralNetwork<IN, OUT> {
        self.optimizer.0.init_with_layers(&self.layers);
        NeuralNetwork::new(
            InputLayer::<IN>,
            self.layers,
            self.error_function.get(),
            self.optimizer.0,
        )
    }
}

/*
macro_rules! impl_add_layer {
    ( $in:path $( ; $name:ident : $fn:ident -> $out:path )+ $(;)? ) => {
        impl NeuralNetworkBuilder<$in> { $(
            pub fn $name(
                mut self,
                neurons: usize,
                act_func: ActivationFunction,
            ) -> NeuralNetworkBuilder<$out> {
                let new_layer = Layer::$fn(self.last_layer.get_neuron_count(), neurons, act_func);
                let neurons = new_layer.get_neuron_count();
                self.layers.push(new_layer);
                NeuralNetworkBuilder {
                    layers: self.layers,
                    last_layer: $out(neurons),
                }
            }
        )+ }
    };
}
*/
