//! # test util module

use crate::{nn::GradComponent, Optimizer, NN};
use const_tensor::{Element, Multidimensional, MultidimensionalOwned, Num, Shape, Tensor};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::iter;

/// [`NN::backprop`] returns the gradient with respect to the input or the output.
///
/// this is for testing
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InspectGrad;

impl fmt::Display for InspectGrad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InspectGrad")
    }
}

impl<X: Element, S: Shape> NN<X, S, S> for InspectGrad {
    type Grad = TensorInspection<X, S>;
    type In = S;
    type OptState<O: Optimizer<X>> = ();
    type StoredData = ();

    fn prop(&self, input: Tensor<X, S>) -> Tensor<X, S> {
        input
    }

    fn train_prop(&self, input: Tensor<X, S>) -> (Tensor<X, S>, Self::StoredData) {
        (input, ())
    }

    fn backprop_inplace(
        &self,
        out_grad: Tensor<X, S>,
        _data: (),
        grad: &mut TensorInspection<X, S>,
    ) {
        grad.tensor = out_grad;
    }

    fn optimize<O: Optimizer<X>>(
        &mut self,
        _grad: &Self::Grad,
        _optimizer: &O,
        _opt_state: &mut (),
    ) {
    }

    fn init_zero_grad(&self) -> Self::Grad {
        unimplemented!()
    }

    fn init_opt_state<O: crate::Optimizer<X>>(&self) -> Self::OptState<O> {
        ()
    }

    fn iter_param(&self) -> impl Iterator<Item = &X> {
        iter::empty()
    }
}

/// Wrapper around a [`Tensor`] which implements [`GradComponent`].
pub struct TensorInspection<X: Element, S: Shape> {
    /// inner [`Tensor`] value
    pub tensor: Tensor<X, S>,
}

impl<X: Num, S: Shape> TensorInspection<X, S> {
    /// creates a new TensorInspection.
    ///
    /// The values of the internal [`Tensor`] are all zero.
    pub fn new() -> Self {
        TensorInspection { tensor: Tensor::zeros() }
    }
}

impl<X: Element, S: Shape> GradComponent<X> for TensorInspection<X, S> {
    fn iter_elem(&self) -> impl Iterator<Item = &X> {
        const_tensor::tensor::iter_elem(&self.tensor)
    }

    fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut X> {
        const_tensor::tensor::iter_elem_mut(&mut self.tensor)
    }
}

impl<X: Element, S: Shape> fmt::Debug for TensorInspection<X, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tensor.fmt(f)
    }
}
