use super::component::{Data, GradComponent, NNComponent};
use const_tensor::{matrix, vector, Element, Len, Matrix, Num, Tensor, TensorData, Vector};

#[derive(Debug)]
pub struct Linear<X: Element, const IN: usize, const OUT: usize, PREV> {
    pub(super) prev: PREV,
    pub(super) weights: Matrix<X, IN, OUT>,
    pub(super) bias: Vector<X, OUT>,
}

impl<X, PREV, const IN: usize, const OUT: usize, NNIN: Tensor<X>>
    NNComponent<X, NNIN, Vector<X, OUT>> for Linear<X, IN, OUT, PREV>
where
    X: Num,
    PREV: NNComponent<X, NNIN, Vector<X, IN>>,
    vector<X, IN>: Len<IN>,
    vector<X, OUT>: Len<OUT>,
    matrix<X, IN, OUT>: Len<{ IN * OUT }>,
    matrix<X, OUT, IN>: Len<{ IN * OUT }>,
{
    type Grad = Linear<X, IN, OUT, PREV::Grad>;
    type In = Vector<X, IN>;
    type StoredData = Data<Vector<X, IN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> const_tensor::Vector<X, OUT> {
        let input = self.prev.prop(input);
        self.weights.mul_vec(&input) + &self.bias
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (Vector<X, OUT>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = self.weights.mul_vec(&input) + &self.bias;
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
    fn backprop(&self, data: Self::StoredData, out_grad: Vector<X, OUT>, grad: &mut Self::Grad) {
        let Data { prev: prev_data, data: input } = data;

        // TODO: bench
        grad.bias.add_elem_mut(&out_grad);
        grad.weights.add_elem_mut(&out_grad.span_mat(&input));
        let input_grad = self.weights.clone().transpose().mul_vec(&out_grad);

        self.prev.backprop(prev_data, input_grad, &mut grad.prev)
    }
}

impl<X: Element, const IN: usize, const OUT: usize, PREV: GradComponent> GradComponent
    for Linear<X, IN, OUT, PREV>
{
}
