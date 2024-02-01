//! # NN module

/*
use crate::trainer::{
    markers::{NoLossFunction, NoOptimizer},
    NNTrainerBuilder,
};
*/
use crate::loss_function::LossFunction;
use const_tensor::{tensor, Element, MultidimArr, Shape, Tensor};
use core::fmt;
use serde::Serialize;
use std::{
    fmt::{Debug, Display},
    iter::Map,
};

pub mod builder;
pub mod component;
mod flatten;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

use crate::trainer::{
    markers::{NoLossFunction, NoOptimizer},
    NNTrainerBuilder,
};
pub use builder::NNBuilder;
pub use component::{NNComponent, NNHead};
pub use flatten::Flatten;
pub use linear::Linear;
pub use relu::{leaky_relu, relu, LeakyReLU, ReLU};
pub use sigmoid::{sigmoid, Sigmoid};
//pub use softmax::{LogSoftmax, Softmax};

/// A neural network.
pub trait NN<X: Element, IN: Shape, OUT: Shape>: NNComponent<X, IN, OUT> + Serialize {
    /// Converts `self` to a [`NNTrainerBuilder`] that can be used to create a [`NNTrainer`]
    ///
    /// Used to
    #[inline]
    fn to_trainer(self) -> NNTrainerBuilder<X, IN, OUT, NoLossFunction, NoOptimizer, Self> {
        NNTrainerBuilder::new(self)
    }

    /// Propagates a [`Tensor`] through the neural network and returns the output [`Tensor`].
    fn propagate(&self, input: &tensor<X, IN>) -> Tensor<X, OUT> {
        // the compiler should inline all prop functions.
        self.prop(input.to_owned())
    }

    /// Iterates over a `batch` of inputs, propagates them and returns an [`Iterator`] over the
    /// outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    fn propagate_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> Tensor<X, OUT> + 'a>
    where
        B: IntoIterator<Item = &'a tensor<X, IN>>,
    {
        batch.into_iter().map(|i| self.propagate(i))
    }

    /// Propagates a [`Tensor`] through the neural network and returns the output [`Tensor`] and
    /// additional data which is required for backpropagation.
    ///
    /// If only the output is needed, use the normal `propagate` method instead.
    fn training_propagate(&self, input: &tensor<X, IN>) -> (Tensor<X, OUT>, Self::StoredData) {
        self.train_prop(input.to_owned())
    }

    /// # Params
    ///
    /// `output_gradient`: gradient of the loss with respect to the network output.
    /// `train_data`: additional data create by `training_propagate` and required for
    /// backpropagation.
    /// `gradient`: stores the changes to each parameter. Has to have the same
    /// dimensions as `self`.
    fn backpropagate(
        &self,
        output_gradient: Tensor<X, OUT>,
        train_data: Self::StoredData,
        mut gradient: Self::Grad,
    ) -> Self::Grad {
        self.backpropagate_inplace(output_gradient, train_data, &mut gradient);
        gradient
    }

    fn backpropagate_inplace(
        &self,
        output_gradient: Tensor<X, OUT>,
        train_data: Self::StoredData,
        gradient: &mut Self::Grad,
    ) {
        self.backprop(output_gradient, train_data, gradient);
    }

    /// Tests the neural network.
    fn test<L: LossFunction<X, OUT>>(
        &self,
        pair: &Pair<X, IN, L::ExpectedOutput>,
        loss_function: &L,
    ) -> TestResult<X, OUT> {
        let (input, expected_output) = pair.as_tuple();
        let out = self.propagate(input);
        let loss = loss_function.propagate(&out, expected_output);
        TestResult::new(out, loss)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// output losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to get the outputs use `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    fn test_batch<'a, B, L, EO: 'a>(
        &'a self,
        batch: B,
        loss_fn: &'a L,
    ) -> Map<B::IntoIter, impl FnMut(&'a Pair<X, IN, EO>) -> TestResult<X, OUT>>
    where
        B: IntoIterator<Item = &'a Pair<X, IN, EO>>,
        L: LossFunction<X, OUT, ExpectedOutput = EO>,
    {
        batch.into_iter().map(|p| self.test(p, loss_fn))
    }

    fn init_zero_gradient(&self) -> Self::Grad {
        self.init_zero_grad()
    }

    /// TODO: lazy
    fn deserialize_hint(&self, deserialized: Self) -> Self {
        deserialized
    }
}

impl<X: Element, IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT> + Serialize> NN<X, IN, OUT>
    for C
{
}

pub struct Pair<X: Element, IN: Shape, EO> {
    input: Tensor<X, IN>,
    expected_output: EO,
}

impl<X: Element, IN: Shape, EO> Pair<X, IN, EO> {
    pub fn new(input: Tensor<X, IN>, expected_output: EO) -> Self {
        Self { input, expected_output }
    }

    pub fn get_input(&self) -> &tensor<X, IN> {
        &self.input
    }

    pub fn get_expected_output(&self) -> &EO {
        &self.expected_output
    }

    pub fn as_tuple(&self) -> (&tensor<X, IN>, &EO) {
        (self.get_input(), self.get_expected_output())
    }

    pub fn into_tuple(self) -> (Tensor<X, IN>, EO) {
        (self.input, self.expected_output)
    }
}

impl<X: Element, IN: Shape, EO> From<(Tensor<X, IN>, EO)> for Pair<X, IN, EO> {
    fn from(value: (Tensor<X, IN>, EO)) -> Self {
        Self::new(value.0, value.1)
    }
}

pub struct TestResult<X: Element, OUT: Shape> {
    output: Tensor<X, OUT>,
    loss: X,
}

impl<X: Element, OUT: Shape> TestResult<X, OUT> {
    pub fn new(output: Tensor<X, OUT>, loss: X) -> Self {
        Self { output, loss }
    }

    pub fn get_output(&self) -> &Tensor<X, OUT> {
        &self.output
    }

    pub fn get_loss(&self) -> X {
        self.loss
    }
}
