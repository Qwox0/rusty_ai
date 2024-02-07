//! # NN module

use crate::{
    loss_function::LossFunction,
    optimizer::Optimizer,
    trainer::{
        markers::{NoLossFunction, NoOptimizer},
        NNTrainerBuilder,
    },
};
use const_tensor::{tensor, Element, MultidimArr, Shape, Tensor};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
    iter::Map,
};

pub mod builder;
pub mod component;
mod flatten;
mod head;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

pub use builder::NNBuilder;
pub use flatten::Flatten;
pub use head::NNHead;
pub use linear::Linear;
pub use relu::{leaky_relu, relu, LeakyReLU, ReLU};
pub use sigmoid::{sigmoid, Sigmoid};
pub use softmax::{LogSoftmax, Softmax};

/// A neural network.
///
/// The neural network is represented by nested component types implementing this trait.
///
/// The innermost type is the first component of the network (usually [`NNHead`]) and the outermost
/// type `Self` is the last component of the network.
///
/// `NNIN`: input of the first component. should be a [`Tensor`].
/// `OUT`: out of this component. should be a [`Tensor`].
pub trait NN<X: Element, NNIN: Shape, OUT: Shape>:
    Sized + fmt::Debug + fmt::Display + PartialEq + Serialize + Send + Sync + 'static
{
    /// Gradient component
    type Grad: GradComponent<X>;

    /// Shape of this components Input tensor.
    ///
    /// currently unused
    type In: Shape;

    /// not the best implementation but has to work for now.
    type OptState<O: Optimizer<X>>: Sized + Send + Sync + 'static;

    /// The data which is saved during `train_prop` and used in `backprop`.
    type StoredData: TrainData;

    /// Propagates the `input` [`Tensor`] through the entire sub network and then through this
    /// component.
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, OUT>;

    /// Like `prop` but also returns the required data for backpropagation.
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, OUT>, Self::StoredData);

    /// Backpropagates the output gradient through this component and then backwards through the
    /// previous components.
    fn backprop_inplace(
        &self,
        out_grad: Tensor<X, OUT>,
        data: Self::StoredData,
        grad: &mut Self::Grad,
    );

    /// Backpropagates the output gradient through this component and then backwards through the
    /// previous components.
    #[inline]
    fn backprop(
        &self,
        out_grad: Tensor<X, OUT>,
        data: Self::StoredData,
        mut grad: Self::Grad,
    ) -> Self::Grad {
        self.backprop_inplace(out_grad, data, &mut grad);
        grad
    }

    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    );

    /// Creates a gradient with the same dimensions as `self` and every element initialized to
    /// `0.0`.
    fn init_zero_grad(&self) -> Self::Grad;
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O>;

    fn iter_param(&self) -> impl Iterator<Item = &X>;

    /// Iterates over a `batch` of inputs, propagates them and returns an [`Iterator`] over the
    /// outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    fn prop_batch<'a, B, I>(
        &'a self,
        batch: B,
    ) -> std::iter::Map<B::IntoIter, impl FnMut(B::Item) -> Tensor<X, OUT> + 'a>
    where
        B: IntoIterator<Item = &'a I>,
        I: ToOwned<Owned = Tensor<X, NNIN>> + 'a,
    {
        batch.into_iter().map(|i: &I| self.prop(i.to_owned()))
    }

    /// Tests the neural network.
    #[inline]
    fn test<L: LossFunction<X, OUT>>(
        &self,
        pair: &Pair<X, NNIN, impl Borrow<L::ExpectedOutput>>,
        loss_function: &L,
    ) -> TestResult<X, OUT> {
        let (input, expected_output) = pair.as_tuple();
        let out = self.prop(input.to_owned());
        let loss = loss_function.propagate(&out, expected_output.borrow());
        TestResult::new(out, loss)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// output losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to get the outputs use `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    fn test_batch<'a, B, L, EO>(
        &'a self,
        batch: B,
        loss_fn: &'a L,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> TestResult<X, OUT>>
    where
        B: IntoIterator<Item = &'a Pair<X, NNIN, EO>>,
        L: LossFunction<X, OUT>,
        EO: Borrow<L::ExpectedOutput> + 'a,
    {
        batch.into_iter().map(|p| self.test(p, loss_fn))
    }

    /// Converts `self` to a [`NNTrainerBuilder`] that can be used to create a [`NNTrainer`]
    ///
    /// Used to
    #[inline]
    fn to_trainer(self) -> NNTrainerBuilder<X, NNIN, OUT, NoLossFunction, NoOptimizer, Self> {
        NNTrainerBuilder::new(self)
    }

    /// TODO: lazy
    #[inline]
    fn deserialize_hint(&self, deserialized: Self) -> Self {
        deserialized
    }
}

macro_rules! component_new {
    ($ty:ident) => {
        component_new! { $ty<PREV> -> }
    };
    ($ty:ident < $( $gen:ident),+ > -> $( $param:ident : $paramty:ty ),* ) => {
        impl<$($gen),+> $ty<$($gen),+> {
            pub fn new( $($param : $paramty ,)* prev: PREV) -> Self {
                $ty { $($param , )* prev }
            }
        }
    };
}
pub(crate) use component_new;

pub trait GradComponent<X: Element>: Sized + Send + Sync + 'static {
    fn set_zero(&mut self);
    fn add_mut(&mut self, other: impl Borrow<Self>);

    fn iter_param(&self) -> impl Iterator<Item = &X>;
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X>;

    fn add(mut self, other: impl Borrow<Self>) -> Self {
        self.add_mut(other);
        self
    }
}

impl<X: Element> GradComponent<X> for () {
    #[inline]
    fn set_zero(&mut self) {}

    #[inline]
    fn add_mut(&mut self, other: impl Borrow<Self>) {}

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        None.into_iter()
    }

    #[inline]
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X> {
        None.into_iter()
    }
}

pub trait TrainData: Sized + Send + Sync + 'static {}

pub struct Data<T, PREV: TrainData> {
    pub prev: PREV,
    pub data: T,
}

impl<T: Sized + Send + Sync + 'static, PREV: TrainData> TrainData for Data<T, PREV> {}

impl TrainData for () {}

/*
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
*/

#[derive(Debug, Clone, Default)]
pub struct Pair<X: Element, IN: Shape, EO> {
    input: Tensor<X, IN>,
    expected_output: EO,
}

impl<X: Element, IN: Shape, EO> Pair<X, IN, EO> {
    pub fn new(input: Tensor<X, IN>, expected_output: EO) -> Self {
        Self { input: input.into(), expected_output: expected_output.into() }
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

impl<X: Element, IN: Shape, EO> From<Pair<X, IN, EO>> for (Tensor<X, IN>, EO) {
    fn from(pair: Pair<X, IN, EO>) -> Self {
        pair.into_tuple()
    }
}

impl<'a, X: Element, IN: Shape, EO> From<&'a Pair<X, IN, EO>> for (&'a tensor<X, IN>, &'a EO) {
    fn from(pair: &'a Pair<X, IN, EO>) -> Self {
        pair.as_tuple()
    }
}

#[derive(Debug, Clone, Default)]
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

    pub fn into_tuple(self) -> (Tensor<X, OUT>, X) {
        (self.output, self.loss)
    }
}

impl<X: Element, OUT: Shape> From<TestResult<X, OUT>> for (Tensor<X, OUT>, X) {
    fn from(res: TestResult<X, OUT>) -> Self {
        res.into_tuple()
    }
}
