use super::component::{Data, GradComponent, NNComponent};
use crate::optimizer::Optimizer;
use const_tensor::{
    Element, Len, Matrix, MatrixShape, Multidimensional, MultidimensionalOwned, Num, Shape, Tensor,
    Vector, VectorShape,
};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;

/// A fully connected layer. Calculates `y = weights * x + bias`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Linear<X: Element, const IN: usize, const OUT: usize, PREV> {
    pub(super) prev: PREV,
    pub(super) weights: Matrix<X, IN, OUT>,
    pub(super) bias: Vector<X, OUT>,
}

impl<X, const IN: usize, const OUT: usize, NNIN, PREV> NNComponent<X, NNIN, [(); OUT]>
    for Linear<X, IN, OUT, PREV>
where
    X: Num,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, [(); IN]>,
{
    type Grad = Linear<X, IN, OUT, PREV::Grad>;
    type In = [(); IN];
    type OptState<O: Optimizer<X>> = LinearOptState<O, X, IN, OUT, PREV::OptState<O>>;
    type StoredData = Data<Vector<X, IN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Vector<X, OUT> {
        let input = self.prev.prop(input);
        self.weights.mul_vec(&input).add_elem(&self.bias)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Vector<X, OUT>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = self.weights.mul_vec(&input).add_vec(&self.bias);
        (out, Data { data: input, prev: prev_data })
    }

    /// ```text
    /// o_i = sum a_j * w_ij over j + b_i
    /// do_i/db_i  = 1
    /// do_i/da_j  = w_ij
    /// do_i/dw_ij = a_j
    ///
    /// dL/db_i  = dL/do_i * 1
    /// dL/da_j  = sum dL/do_i * w_ij over i
    /// dL/dw_ij = dL/do_i * a_j
    ///
    /// -----
    ///
    /// o_i: output component i
    /// a_i: input component i
    /// w_ij: weight from a_j to o_i
    /// b_i: bias component i
    ///
    /// L: total loss
    /// ```
    #[inline]
    fn backprop(&self, out_grad: Vector<X, OUT>, data: Self::StoredData, grad: &mut Self::Grad) {
        let Data { prev: prev_data, data: input } = data;

        // TODO: bench
        grad.bias.add_elem_mut(&out_grad);
        grad.weights.add_elem_mut(&out_grad.span_mat(&input));
        let input_grad = self.weights.clone().transpose().mul_vec(&out_grad);

        self.prev.backprop(input_grad, prev_data, &mut grad.prev)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        state: &mut Self::OptState<O>,
    ) {
        optimizer.optimize_tensor(&mut self.weights, &grad.weights, &mut state.weights);
        optimizer.optimize_tensor(&mut self.bias, &grad.bias, &mut state.bias);
        self.prev.optimize(&grad.prev, optimizer, &mut state.prev);
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        let prev = self.prev.init_zero_grad();
        Linear { prev, weights: Matrix::zeros(), bias: Vector::zeros() }
    }

    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        LinearOptState {
            prev: self.prev.init_opt_state(),
            weights: O::new_state(Matrix::zeros()),
            bias: O::new_state(Vector::zeros()),
        }
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev
            .iter_param()
            .chain(self.weights.iter_elem().chain(self.bias.iter_elem()))
    }
}

pub struct LinearOptState<O: Optimizer<X>, X: Element, const IN: usize, const OUT: usize, PREV> {
    pub(super) prev: PREV,
    pub(super) weights: O::State<MatrixShape<IN, OUT>>,
    pub(super) bias: O::State<VectorShape<OUT>>,
}

/*
impl<X: Element + Serialize, const IN: usize, const OUT: usize, PREV: Serialize> Serialize
    for Linear<X, IN, OUT, PREV>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut s = serializer.serialize_struct(std::any::type_name::<Self>(), 3)?;
        s.serialize_field("prev", &self.prev)?;
        s.serialize_field("weights", &self.weights)?;
        s.serialize_field("bias", &self.bias)?;
        s.end()
    }
}
*/

impl<'a, X, const IN: usize, const OUT: usize, PREV> fmt::Display for Linear<X, IN, OUT, PREV>
where
    X: Element,
    PREV: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        write!(f, "Linear: {IN} -> {OUT}")
    }
}

impl<X: Num, const IN: usize, const OUT: usize, PREV: GradComponent<X>> GradComponent<X>
    for Linear<X, IN, OUT, PREV>
{
    fn set_zero(&mut self) {
        self.weights.fill_zero();
        self.bias.fill_zero();
        self.prev.set_zero();
    }

    #[inline]
    fn add_mut(&mut self, other: impl Borrow<Self>) {
        let other = other.borrow();
        self.weights.add_elem_mut(&other.weights);
        self.bias.add_elem_mut(&other.bias);
        self.prev.add_mut(&other.prev);
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev
            .iter_param()
            .chain(self.weights.iter_elem())
            .chain(self.bias.iter_elem())
    }

    #[inline]
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X> {
        self.prev
            .iter_param_mut()
            .chain(self.weights.iter_elem_mut().chain(self.bias.iter_elem_mut()))
    }
}
