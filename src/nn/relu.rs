use super::component::{component_new, Data, NNComponent, NNDisplay};
use crate::optimizer::Optimizer;
use const_tensor::{Element, Len, Num, Shape, Tensor, TensorData};
use core::fmt;
use serde::{Deserialize, Serialize};

#[inline]
pub fn relu<X: Num>(x: X) -> X {
    if x.is_positive() { x } else { X::ZERO }
}

#[inline]
pub fn leaky_relu<X: Num>(x: X, leak_rate: X) -> X {
    if x.is_positive() { x } else { leak_rate }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReLU<PREV> {
    pub(super) prev: PREV,
}

component_new! { ReLU }

impl<X, S, NNIN, PREV> NNComponent<X, NNIN, S> for ReLU<PREV>
where
    X: Num,
    S: Shape + Len<{ S::LEN }>,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<Tensor<bool, S>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, S> {
        let input = self.prev.prop(input);
        input.map_inplace(relu)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, S>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_inplace(relu);
        let data = out.map_clone(|x| !x.is_zero());
        (out, Data { data, prev: prev_data })
    }

    /// ```
    /// o_i = relu(a_i)
    /// do_i/da_i = relu'(a_i)
    /// dL/da_i   = dL/do_i * relu'(a_i)
    ///
    /// -----
    ///
    /// o_i: output component i
    /// a_i: input component i
    ///
    /// L: total loss
    /// ```
    #[inline]
    fn backprop(&self, out_grad: Tensor<X, S>, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= X::from_bool(is_pos);
        }
        self.prev.backprop(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        mut opt_state: Self::OptState<O>,
    ) -> Self::OptState<O> {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev.iter_param()
    }
}

impl<'a, PREV> fmt::Display for NNDisplay<'a, ReLU<PREV>>
where NNDisplay<'a, PREV>: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", NNDisplay(&self.0.prev))?;
        write!(f, "ReLU")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakyReLU<X, PREV> {
    pub(super) prev: PREV,
    pub(super) leak_rate: X,
}

component_new! { LeakyReLU<X, PREV> -> leak_rate: X }

impl<X, S, NNIN, PREV> NNComponent<X, NNIN, S> for LeakyReLU<X, PREV>
where
    X: Num,
    S: Shape + Len<{ S::LEN }>,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<Tensor<bool, S>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, S> {
        let input = self.prev.prop(input);
        input.map_inplace(|x| leaky_relu(x, self.leak_rate))
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, S>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_inplace(|x| leaky_relu(x, self.leak_rate));
        let data = out.map_clone(|x| !x.is_zero());
        (out, Data { data, prev: prev_data })
    }

    #[inline]
    fn backprop(&self, out_grad: Tensor<X, S>, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= if is_pos { X::ONE } else { self.leak_rate }
        }
        self.prev.backprop(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        mut opt_state: Self::OptState<O>,
    ) -> Self::OptState<O> {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev.iter_param()
    }
}

impl<'a, X: Element, PREV> fmt::Display for NNDisplay<'a, LeakyReLU<X, PREV>>
where NNDisplay<'a, PREV>: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", NNDisplay(&self.0.prev))?;
        write!(f, "LeakyReLU (leak_rate: {:?})", self.0.leak_rate)
    }
}
