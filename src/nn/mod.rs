//! # NN module

use crate::{
    loss_function::LossFunction,
    optimizer::Optimizer,
    test_result::TestResult,
    trainer::{
        markers::{NoLossFunction, NoOptimizer},
        NNTrainerBuilder,
    },
    Pair,
};
use const_tensor::{Element, Num, Shape, Tensor};
use core::fmt;
use serde::Serialize;
use std::{borrow::Borrow, iter::Map};

pub mod builder;
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
/// `NNIN`: input [`Shape`] of the first component.
/// `OUT`: output [`Shape`] of this component.
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

    /// Optimizes the nn/nn component.
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    );

    /// Creates a gradient with the same dimensions as `self` and every element initialized to
    /// `0.0`.
    fn init_zero_grad(&self) -> Self::Grad;

    /// Creates a new State for the [`Optimizer`] `O`.
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O>;

    /// Creates an [`Iterator`] over the parameters of `self`.
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
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> Tensor<X, OUT> + 'a>
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
            /// Creates a new nn component.
            pub fn new( $($param : $paramty ,)* prev: PREV) -> Self {
                $ty { $($param , )* prev }
            }
        }
    };
}
pub(crate) use component_new;

/// see [`const_tensor::Multidimensional`]
pub trait GradComponent<X: Element>: Sized + Send + Sync + 'static {
    /// see [`const_tensor::Multidimensional`]
    fn iter_elem(&self) -> impl Iterator<Item = &X>;
    /// see [`const_tensor::Multidimensional`]
    fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut X>;

    /// see [`const_tensor::Multidimensional`]
    fn fill_zero(&mut self)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x = X::ZERO;
        }
    }

    /// see [`const_tensor::Multidimensional`]
    fn add_elem_mut(&mut self, other: impl Borrow<Self>)
    where X: Num {
        for (x, y) in self.iter_elem_mut().zip(other.borrow().iter_elem()) {
            *x += *y;
        }
    }

    /// see [`const_tensor::Multidimensional`]
    fn add_elem(mut self, other: impl Borrow<Self>) -> Self
    where X: Num {
        self.add_elem_mut(other);
        self
    }
}

/// Marker for the data calculated in [`NN::train_prop`].
pub trait TrainData: Sized + Send + Sync + 'static {}

/// Nested type for storing the data calculated in [`NN::train_prop`].
pub struct Data<T, PREV: TrainData> {
    /// stored data for the previous components
    pub prev: PREV,
    /// stored data for this component.
    pub data: T,
}

impl<T: Sized + Send + Sync + 'static, PREV: TrainData> TrainData for Data<T, PREV> {}
