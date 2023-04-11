use std::{iter::once, marker::PhantomData};

use itertools::Itertools;

use crate::{activation_function::ActivationFunction, layer::Layer, neural_network::NeuralNetwork};

// Markers
pub struct NoLayer;
pub struct Input<const N: usize>;
pub struct Hidden;
pub struct Output<const N: usize>;

pub trait InputOrHidden {}
impl<const N: usize> InputOrHidden for Input<N> {}
impl InputOrHidden for Hidden {}
// Markers end

#[derive(Debug)]
pub struct NeuralNetworkBuilder<IN, LAST> {
    layers: Vec<Layer>,
    input: PhantomData<IN>,
    last_layer: PhantomData<LAST>,
}

impl NeuralNetworkBuilder<NoLayer, NoLayer> {
    pub fn new() -> NeuralNetworkBuilder<NoLayer, NoLayer> {
        NeuralNetworkBuilder {
            layers: vec![],
            input: PhantomData,
            last_layer: PhantomData,
        }
    }

    pub fn input_layer<const N: usize>(self) -> NeuralNetworkBuilder<Input<N>, Input<N>> {
        NeuralNetworkBuilder {
            layers: vec![Layer::new_input(N)],
            input: PhantomData,
            last_layer: PhantomData,
        }
    }
}

impl<T: InputOrHidden, const N: usize> NeuralNetworkBuilder<Input<N>, T> {
    pub fn hidden_layer(
        mut self,
        neurons: usize,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Hidden> {
        self.layers
            .push(Layer::new_hidden(self.last_count(), neurons, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
        }
    }

    pub fn hidden_layers(
        mut self,
        neurons_per_layer: &[usize],
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Hidden> {
        let last_count = self.last_count();
        for (last, count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers.push(Layer::new_hidden(*last, *count, act_func));
        }
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
        }
    }

    pub fn output_layer<const OUT: usize>(
        mut self,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Input<N>, Output<OUT>> {
        self.layers
            .push(Layer::new_output(self.last_count(), OUT, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
            input: PhantomData,
            last_layer: PhantomData,
        }
    }

    fn last_count(&self) -> usize {
        self.layers
            .last()
            .expect("self.layers always contains an InputLayer")
            .get_neuron_count()
    }
}

impl<const IN: usize, const OUT: usize> NeuralNetworkBuilder<Input<IN>, Output<OUT>> {
    pub fn build(self) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork::new(self.layers)
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
