use super::component::{Data, GradComponent, NNComponent, NNDisplay};
use const_tensor::{Element, Len, Matrix, Num, Shape, Tensor, Vector};
use core::fmt;

#[derive(Debug)]
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
    [(); IN]: Len<IN>,
    [(); OUT]: Len<OUT>,
    [[(); IN]; OUT]: Len<{ IN * OUT }>,
    [[(); OUT]; IN]: Len<{ IN * OUT }>,
{
    type Grad = Linear<X, IN, OUT, PREV::Grad>;
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

    /// ```
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
}

impl<'a, X, const IN: usize, const OUT: usize, PREV> fmt::Display
    for NNDisplay<'a, Linear<X, IN, OUT, PREV>>
where
    X: Element,
    NNDisplay<'a, PREV>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", NNDisplay(&self.0.prev))?;
        write!(f, "Linear: {IN} -> {OUT}")
    }
}

impl<X: Element, const IN: usize, const OUT: usize, PREV: GradComponent> GradComponent
    for Linear<X, IN, OUT, PREV>
{
}
