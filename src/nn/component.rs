use crate::{loss_function::LossFunction, optimizer::Optimizer, trainer::{markers::{NoLossFunction, NoOptimizer}, NNTrainerBuilder}};
#[allow(unused_imports)]
use const_tensor::Tensor;
use const_tensor::{Element, Shape};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, iter::Map};


/// Helper trait for components which aren't affected by training. (activation functions, ...)
pub trait NoTrainComponent<X: Element, IN, OUT, PREV> {
    type StoredData;

    fn get_prev(&self) -> &PREV;
    fn get_prev_mut(&mut self) -> &mut PREV;

    fn prop(&self, input: IN) -> OUT;

    fn train_prop(&self, input: IN) -> (OUT, Self::StoredData);

    fn backprop(&self, out_grad: OUT, data: Self::StoredData) -> IN;
}

/// Implements [`NNComponent`] for types implementing [`NoTrainComponent`].
#[macro_export]
macro_rules! derive_nn_component {
    ($ty:ty : $in:ident -> $out:ident) => {
        impl<X, NNIN, $in, $out, PREV> NNComponent<X, NNIN, $out> for $ty
        where
            Self: NoTrainComponent<X, $in, $out, PREV>,
            X: Element,
            PREV: NNComponent<X, NNIN, $in>,
        {
            type Grad = PREV::Grad;
            type OptState<O: Optimizer<X>> = PREV::OptState<O>;
            type StoredData = Data<Self::StoredData, PREV::StoredData>;

            #[inline]
            fn prop(&self, input: NNIN) -> $out {
                let input = self.get_prev().prop(input);
                NoTrainComponent::prop(&self, input)
            }

            #[inline]
            fn train_prop(&self, input: NNIN) -> ($out, Self::StoredData) {
                let (input, prev_data) = self.get_prev().train_prop(input);
                let (out, data) = self.train_prop(input);
                (out, Data { data, prev: prev_data })
            }

            #[inline]
            fn backprop(&self, out_grad: $out, data: Self::StoredData, grad: &mut PREV::Grad) {
                let Data { prev: prev_data, data } = data;
                let input_grad = self.backprop(out_grad, data);
                self.prev.backprop(input_grad, prev_data, grad)
            }

            #[inline]
            fn optimize<O: Optimizer<X>>(
                &mut self,
                grad: &PREV::Grad,
                optimizer: &O,
                opt_state: &mut PREV::OptState<O>,
            ) {
                self.get_prev_mut().optimize(grad, optimizer, opt_state);
            }

            #[inline]
            fn init_zero_grad(&self) -> PREV::Grad {
                self.get_prev().init_zero_grad()
            }

            #[inline]
            fn init_opt_state<O: Optimizer<X>>(&self) -> PREV::OptState<O> {
                self.get_prev().init_opt_state()
            }

            #[inline]
            fn iter_param(&self) -> impl Iterator<Item = &X> {
                self.get_prev().iter_param()
            }
        }
    };
}
use super::{Pair, TestResult};
pub use derive_nn_component;
