use std::{iter::once, marker::PhantomData};

use itertools::Itertools;

use crate::{
    activation_function::ActivationFunction::{self, *},
    layer::Layer,
    neural_network::NeuralNetwork,
};

// Markers
pub struct None;
pub struct Input;
pub struct Hidden;
pub struct Output;

pub trait InputOrHidden {}
impl InputOrHidden for Input {}
impl InputOrHidden for Hidden {}
// Markers end

pub struct NeuralNetworkBuilder<T> {
    layers: Vec<Layer>,
    last_layer: PhantomData<T>,
}

impl NeuralNetworkBuilder<None> {
    pub fn input_layer(neurons: usize) -> NeuralNetworkBuilder<Input> {
        NeuralNetworkBuilder {
            layers: vec![Layer::new_input(neurons)],
            last_layer: PhantomData,
        }
    }
}

impl<T: InputOrHidden> NeuralNetworkBuilder<T> {
    pub fn hidden_layer(
        mut self,
        neurons: usize,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Hidden> {
        self.layers.push(Layer::new_hidden(self.last_count(), neurons, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
            last_layer: PhantomData,
        }
    }

    pub fn hidden_layers(
        mut self,
        neurons_per_layer: &[usize],
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Hidden> {
        let last_count = self.last_count();
        for (last, count) in once(&last_count).chain(neurons_per_layer).tuple_windows() {
            self.layers.push(Layer::new_hidden(*last, *count, act_func));
        }
        NeuralNetworkBuilder {
            layers: self.layers,
            last_layer: PhantomData,
        }
    }

    pub fn output_layer(
        mut self,
        neurons: usize,
        act_func: ActivationFunction,
    ) -> NeuralNetworkBuilder<Output> {
        self.layers
            .push(Layer::new_output(self.last_count(), neurons, act_func));
        NeuralNetworkBuilder {
            layers: self.layers,
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

impl NeuralNetworkBuilder<Output> {
    pub fn build(self) -> NeuralNetwork {
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
