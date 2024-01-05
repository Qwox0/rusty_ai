use super::component::{Data, GradComponent, NNComponent};
use const_tensor::{vector, Element, Len, Matrix, Num, Tensor, Vector};

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
{
    type Grad = Linear<X, IN, OUT, PREV::Grad>;
    type In = Vector<X, IN>;
    type StoredData = Data<Vector<X, IN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> const_tensor::Vector<X, OUT> {
        let input = self.prev.prop(input);
        self.weights.mul_vec(&input) + &self.bias
    }

    fn train_prop(&self, input: NNIN) -> (Vector<X, OUT>, Self::StoredData) {
        let (input, a) = self.prev.train_prop(input);
        let out = self.weights.mul_vec(&input) + &self.bias;
        (out, Data { data: input, prev: a })
    }

    #[inline]
    fn backprop(&self, data: Self::StoredData, out_grad: Vector<X, OUT>, grad: &mut Self::Grad) {
        todo!()
    }
}

impl<X: Element, const IN: usize, const OUT: usize, PREV: GradComponent> GradComponent
    for Linear<X, IN, OUT, PREV>
{
}
