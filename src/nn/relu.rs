use super::{component_new, Data, NN};
use crate::optimizer::Optimizer;
use const_tensor::{Element, Len, Multidimensional, MultidimensionalOwned, Num, Shape, Tensor};
use core::fmt;
use serde::{Deserialize, Serialize};

/// see [`ReLU`].
#[inline]
pub fn relu<X: Num>(x: X) -> X {
    if x.is_positive() { x } else { X::ZERO }
}

/// see [`LeakyReLU`].
#[inline]
pub fn leaky_relu<X: Num>(x: X, leak_rate: X) -> X {
    if x.is_positive() { x } else { leak_rate * x }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReLU<PREV> {
    pub(super) prev: PREV,
}

component_new! { ReLU }

impl<X, S, NNIN, PREV> NN<X, NNIN, S> for ReLU<PREV>
where
    X: Num,
    S: Shape,
    NNIN: Shape,
    PREV: NN<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type In = S;
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements are positive.
    type StoredData = Data<Tensor<bool, S>, PREV::StoredData>;

    /// # Examples
    ///
    /// ```rust
    /// # use rusty_ai::{*, nn::*, const_tensor::*};
    /// let relu = ReLU::new(NNHead);
    /// let out = relu.prop(Vector::new([-1.0, 0.0, 1.0]));
    /// assert_eq!(out, [0.0, 0.0, 1.0]);
    /// ```
    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, S> {
        let input = self.prev.prop(input);
        input.map_inplace(relu)
    }

    /// # Examples
    ///
    /// ```rust
    /// # use rusty_ai::{*, nn::*, const_tensor::*};
    /// let relu = ReLU::new(NNHead);
    /// let (out, data) = relu.train_prop(Vector::new([-1.0, 0.0, 1.0]));
    /// assert_eq!(out, [0.0, 0.0, 1.0]);
    /// assert_eq!(data.data, [false, false, true]);
    /// ```
    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, S>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_inplace(relu);
        let data = out.map_clone(|x| x > X::ZERO);
        (out, Data { data, prev: prev_data })
    }

    /// ```text
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rusty_ai::{*, nn::*, const_tensor::*};
    /// let relu = ReLU::new(NNHead);
    /// let (_, data) = relu.train_prop(Vector::new([-1.0, 0.0, 1.0]));
    /// let out_grad = Tensor::new([1.0, 1.0, 1.0]);
    /// let in_grad = relu.backprop(out_grad, data, ());
    /// panic!("{:?}", in_grad);
    /// ```
    #[inline]
    fn backprop_inplace(&self, out_grad: Tensor<X, S>, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= X::from_bool(is_pos);
        }
        self.prev.backprop_inplace(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    ) {
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

impl<PREV> fmt::Display for ReLU<PREV>
where PREV: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        write!(f, "ReLU")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakyReLU<X, PREV> {
    pub(super) prev: PREV,
    pub(super) leak_rate: X,
}

component_new! { LeakyReLU<X, PREV> -> leak_rate: X }

impl<X, S, NNIN, PREV> NN<X, NNIN, S> for LeakyReLU<X, PREV>
where
    X: Num,
    S: Shape,
    NNIN: Shape,
    PREV: NN<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type In = S;
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
        let data = out.map_clone(|x| x > X::ZERO);
        (out, Data { data, prev: prev_data })
    }

    #[inline]
    fn backprop_inplace(&self, out_grad: Tensor<X, S>, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= if is_pos { X::ONE } else { self.leak_rate }
        }
        self.prev.backprop_inplace(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    ) {
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

impl<X: Element, PREV> fmt::Display for LeakyReLU<X, PREV>
where PREV: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        write!(f, "LeakyReLU (leak_rate: {:?})", self.leak_rate)
    }
}
