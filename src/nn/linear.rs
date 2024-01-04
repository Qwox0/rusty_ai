use super::component::NNComponent;
use const_tensor::{vector, Element, Len, Matrix, Num, Tensor, Tensor3, TensorData, Vector};

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
    #[inline]
    fn prop(&self, input: NNIN) -> const_tensor::Vector<X, OUT> {
        let input = self.prev.prop(input);
        self.weights.mul_vec(input) + &self.bias
    }

    #[inline]
    fn backprop(&mut self) {
        todo!()
    }
}
