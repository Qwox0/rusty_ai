use super::component::{Data, NNComponent, NNDisplay};
use const_tensor::{Float, Len, Num, Shape, Tensor, TensorData};
use core::fmt;

#[inline]
pub fn sigmoid<X: Float>(x: X) -> X {
    x.neg().exp().add(X::ONE).recip()
}

#[derive(Debug)]
pub struct Sigmoid<PREV> {
    pub(super) prev: PREV,
}

impl<X, S, NNIN, PREV> NNComponent<X, NNIN, S> for Sigmoid<PREV>
where
    X: Float,
    S: Shape + Len<{ S::LEN }>,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<Tensor<X, S>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, S> {
        let input = self.prev.prop(input);
        input.map_inplace(sigmoid)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, S>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_inplace(sigmoid);
        (out.clone(), Data { data: out, prev: prev_data })
    }

    /// ```
    /// o_i = sigmoid(a_i)
    /// do_i/da_i = sigmoid'(a_i) = sigmoid(a_i) * (1 - sigmoid(a_i))
    /// dL/da_i   = dL/do_i * sigmoid(a_i) * (1 - sigmoid(a_i))
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
        let Data { prev: prev_data, data: output } = data;
        let input_grad = out_grad.mul_elem(&output.map_inplace(|x| x * (X::ONE - x)));
        self.prev.backprop(input_grad, prev_data, grad)
    }
}

impl<'a, PREV> fmt::Display for NNDisplay<'a, Sigmoid<PREV>>
where NNDisplay<'a, PREV>: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", NNDisplay(&self.0.prev))?;
        write!(f, "Sigmoid")
    }
}
