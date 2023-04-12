use std::{iter::once, marker::PhantomData};

use itertools::Itertools;

use crate::{
    activation_function::ActivationFunction,
    layer::Layer,
    neural_network::NeuralNetwork,
    optimizer::{Optimizer, OptimizerDispatch},
};

// Markers
pub struct NoLayer;
pub struct Input<const N: usize>;
pub struct Hidden;
pub struct Output<const N: usize>;

pub trait InputOrHidden {}
impl<const N: usize> InputOrHidden for Input<N> {}
impl InputOrHidden for Hidden {}

pub struct NoOptimizer;
pub struct HasOptimizer(OptimizerDispatch);
// Markers end

#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST, OPTIMIZER> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
    optimizer: OPTIMIZER,
}

impl NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer> {
    pub fn new() -> NeuralNetworkBuilder<NoLayer, NoLayer, NoOptimizer> {
        NeuralNetworkBuilder {
            layers: vec![],
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: NoOptimizer,
        }
    }
}
impl<O> NeuralNetworkBuilder<NoLayer, NoLayer, O> {
    pub fn input_layer<const N: usize>(self) -> NeuralNetworkBuilder<Input<N>, Input<N>, O> {
        NeuralNetworkBuilder {
            layers: vec![Layer::new_input(N)],
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: self.optimizer,
        }
    }
}

impl<T: InputOrHidden, O, const N: usize> NeuralNetworkBuilder<Input<N>, T, O> {
    pub fn hidden_layer(
        mut self,
        neurons: usize,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Hidden, O> {
        self.layers
            .push(Layer::new_hidden(self.last_count(), neurons, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: self.optimizer,
        }
    }

    pub fn hidden_layers(
        mut self,
        neurons_per_layer: &[usize],
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Hidden, O> {
        let last_count = self.last_count();
        for (last, count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers.push(Layer::new_hidden(*last, *count, act_func));
        }
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: self.optimizer,
        }
    }

    pub fn output_layer<const OUT: usize>(
        mut self,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Output<OUT>, O> {
        self.layers
            .push(Layer::new_output(self.last_count(), OUT, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: self.optimizer,
        }
    }

    fn last_count(&self) -> usize {
        self.layers
            .last()
            .expect("self.layers always contains an InputLayer")
            .get_neuron_count()
    }
}

impl<const IN: usize, const OUT: usize> NeuralNetworkBuilder<Input<IN>, Output<OUT>, NoOptimizer> {
    pub fn optimizer(
        self,
        optimizer: OptimizerDispatch,
    ) -> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer> {
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
            optimizer: HasOptimizer(optimizer),
        }
    }
}

impl<const IN: usize, const OUT: usize> NeuralNetworkBuilder<Input<IN>, Output<OUT>, HasOptimizer> {
    pub fn build(mut self) -> NeuralNetwork<IN, OUT> {
        self.optimizer.0.init_with_layers(&self.layers);
        NeuralNetwork::new(self.layers, self.optimizer.0)
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
