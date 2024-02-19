use super::GradComponent;
use crate::{nn::TrainData, Optimizer, NN};
use const_tensor::{Element, Shape, Tensor};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::iter;

/// The start of the nested nn component type.
#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NNHead;

impl<X: Element, NNIN: Shape> NN<X, NNIN, NNIN> for NNHead {
    type Grad = ();
    type In = NNIN;
    type OptState<O: Optimizer<X>> = ();
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
    fn backprop_inplace(&self, _out_grad: Tensor<X, NNIN>, _data: (), _grad: &mut ()) {}

    #[inline]
    fn optimize<O: Optimizer<X>>(&mut self, _grad: &(), _optimizer: &O, _opt_state: &mut ()) {}

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {}

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {}

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        iter::empty()
    }
}

impl fmt::Display for NNHead {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<X: Element> GradComponent<X> for () {
    fn iter_elem(&self) -> impl Iterator<Item = &X> {
        iter::empty()
    }

    fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut X> {
        iter::empty()
    }
}

impl TrainData for () {}
