//! # NN module

#[allow(unused_imports)]
use crate::trainer::NNTrainer;
use crate::{
    gradient::aliases::OutputGradient,
    layer::Layer,
    matrix::Matrix,
    prelude::LossFunction,
    trainer::{
        markers::{NoLossFunction, NoOptimizer},
        NNTrainerBuilder,
    },
    Gradient, Input, ParamsIter, VerbosePropagation,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, iter::Map};

pub mod builder;
pub use builder::{BuildLayer, NNBuilder};

/// layers contains all Hidden Layers and the Output Layers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
}

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// use [`NNBuilder`] instead!
    #[inline]
    fn new(layers: Vec<Layer>) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork { layers }
    }

    /// Converts `self` to a [`NNTrainerBuilder`] that can be used to create a [`NNTrainer`]
    ///
    /// Used to
    #[inline]
    pub fn to_trainer(self) -> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
        NNTrainerBuilder::new(self)
    }

    /// Returns the layers of `self` as a slice.
    #[inline]
    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Creates a [`Gradient`] with the same dimensions as `self` and every element initialized to
    /// `0.0`
    pub fn init_zero_gradient(&self) -> Gradient {
        self.layers.iter().map(Layer::init_zero_gradient).collect()
    }

    /// Propagates an [`Input`] through the neural network and returns its output.
    pub fn propagate(&self, input: &Input<IN>) -> [f64; OUT] {
        let input = Cow::from(input.as_slice());
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.propagate(&acc).into())
            .into_owned()
            .try_into()
            .expect("last layer should have `OUT` neurons")
    }

    /// Iterates over a `batch` of inputs, propagates them and returns an [`Iterator`] over the
    /// outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn propagate_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> [f64; OUT] + 'a>
    where
        B: IntoIterator<Item = &'a Input<IN>>,
    {
        batch.into_iter().map(|i| self.propagate(i))
    }

    /// Propagates an [`Input`] through the neural network and returns the input and the outputs of
    /// every layer.
    ///
    /// If only the final output is needed, use `propagate` instead.
    ///
    /// This is used internally during training.
    pub fn verbose_propagate(&self, input: &Input<IN>) -> VerbosePropagation<OUT> {
        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        let nn_out = self.layers.iter().fold(input.to_vec(), |input, layer| {
            let out = layer.propagate(&input);
            outputs.push(input);
            out
        });
        outputs.push(nn_out);
        VerbosePropagation::new(outputs)
    }

    /// # Params
    ///
    /// `verbose_prop`: input and activation of every layer
    /// `nn_output_gradient`: gradient of the loss with respect to the network output (input to
    /// loss function)
    /// `gradient`: stores the changes to each parameter. Has to have the same dimensions as
    /// `self`.
    pub fn backpropagate(
        &self,
        verbose_prop: &VerbosePropagation<OUT>,
        nn_output_gradient: OutputGradient,
        gradient: &mut Gradient,
    ) {
        self.layers
            .iter()
            .zip(&mut gradient.layers)
            .zip(verbose_prop.iter_layers())
            .rev()
            .fold(nn_output_gradient, |output_gradient, ((layer, gradient), [input, output])| {
                layer.backpropagate(input, output, output_gradient, gradient)
            });
    }

    /// Tests the neural network.
    pub fn test<L: LossFunction<OUT>>(
        &self,
        input: &Input<IN>,
        expected_output: &L::ExpectedOutput,
        loss_function: &L,
    ) -> ([f64; OUT], f64) {
        let out = self.propagate(input);
        let loss = loss_function.propagate(&out, expected_output);
        (out, loss)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// output losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to get the outputs use `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    pub fn test_batch<'a, B, L, EO: 'a>(
        &'a self,
        batch: B,
        loss_fn: &'a L,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> ([f64; OUT], f64) + 'a>
    where
        B: IntoIterator<Item = (&'a Input<IN>, &'a EO)>,
        L: LossFunction<OUT, ExpectedOutput = EO>,
    {
        batch.into_iter().map(|(input, eo)| self.test(input, eo, loss_fn))
    }
}

impl<const IN: usize, const OUT: usize> ParamsIter for NeuralNetwork<IN, OUT> {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a f64> {
        self.layers.iter().map(Layer::iter).flatten()
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut f64> {
        self.layers.iter_mut().map(Layer::iter_mut).flatten()
    }
}

impl<'a, const IN: usize, const OUT: usize> Default for NeuralNetwork<IN, OUT> {
    /// creates a [`NeuralNetwork`] with one layer. Every parameter is equal to `0.0`.
    ///
    /// This is probably only useful for testing.
    fn default() -> Self {
        NNBuilder::default()
            .input()
            .layer_from_parameters(Matrix::with_zeros(IN, OUT), vec![0.0; OUT].into())
            .build()
    }
}

impl<const IN: usize, const OUT: usize> std::fmt::Display for NeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let get_plural_s = |x: usize| if x == 1 { "" } else { "s" };
        writeln!(
            f,
            "Neural Network: {IN} Input{} -> {OUT} Output{}",
            get_plural_s(IN),
            get_plural_s(OUT),
        )?;
        let layers_text =
            self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", layers_text)
    }
}