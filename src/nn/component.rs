use const_tensor::{Element, Shape, Tensor};
use core::fmt;
use std::{
    fmt::Debug,
    marker::{PhantomData, PhantomPinned},
};

pub trait NNComponent<X: Element, NNIN: Shape, OUT: Shape>: Sized + fmt::Debug {
    /// Gradient component
    type Grad: GradComponent;
    // /// Shape of this components Input tensor.
    // type In: Shape;
    /// The data which is saved during `train_prop` and used in `backprop`.
    type StoredData: TrainData;

    /// Propagates the `input` [`Tensor`] through the entire sub network and then through this
    /// component.
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, OUT>;

    /// Like `prop` but also returns the required data for backpropagation.
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, OUT>, Self::StoredData);

    /// Backpropagates the output gradient through this component and then backwards through the
    /// previous components.
    fn backprop(&self, out_grad: Tensor<X, OUT>, data: Self::StoredData, grad: &mut Self::Grad);
}

pub trait GradComponent {}

pub trait TrainData {}

pub struct Data<T, PREV: TrainData> {
    pub prev: PREV,
    pub data: T,
}

impl<T, PREV: TrainData> TrainData for Data<T, PREV> {}

/// Wrapper for [`NNComponent`] to implement [`fmt::Display`].
pub struct NNDisplay<'a, C>(pub &'a C);

impl<X: Element, NNIN: Shape> NNComponent<X, NNIN, NNIN> for () {
    type Grad = ();
    type StoredData = ();

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, NNIN> {
        input
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, NNIN>, Self::StoredData) {
        (input, ())
    }

    #[inline]
    fn backprop(&self, _out_grad: Tensor<X, NNIN>, _data: (), _grad: &mut ()) {}
}

impl fmt::Display for NNDisplay<'_, ()> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl GradComponent for () {}

impl TrainData for () {}
