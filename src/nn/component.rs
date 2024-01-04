use super::{linear::Linear, relu::ReLU, NN};
use const_tensor::{Element, Matrix, Tensor, Vector};
use std::{fmt::Debug, marker::PhantomData};

pub trait NNComponent<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>>: Debug + Sized {
    fn prop(&self, input: NNIN) -> OUT;

    fn backprop(&mut self);
}

impl<X: Element, NNIN: Tensor<X>> NNComponent<X, NNIN, NNIN> for () {
    fn prop(&self, input: NNIN) -> NNIN {
        input
    }

    fn backprop(&mut self) {
        todo!()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Builder<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>, C> {
    components: C,
    _out: PhantomData<OUT>,
    _marker: PhantomData<(X, NNIN)>,
}

impl<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>, PREV: NNComponent<X, NNIN, OUT>>
    Builder<X, NNIN, OUT, PREV>
{
    #[inline]
    fn add_component<C: NNComponent<X, NNIN, OUT2>, OUT2: Tensor<X>>(
        self,
        c: impl FnOnce(PREV) -> C,
    ) -> Builder<X, NNIN, OUT2, C> {
        Builder { components: c(self.components), _out: PhantomData, ..self }
    }

    fn relu(self) -> Builder<X, NNIN, OUT, ReLU<PREV>> {
        Builder { components: ReLU { prev: self.components }, ..self }
    }

    fn build(self) -> NN<X, NNIN, OUT, PREV> {
        NN { components: self.components, _marker: PhantomData }
    }
}

impl<X: Element, NNIN: Tensor<X>, const IN: usize, PREV: NNComponent<X, NNIN, Vector<X, IN>>>
    Builder<X, NNIN, Vector<X, IN>, PREV>
{
    fn layer<const N: usize>(
        self,
        weights: [[X; IN]; N],
        bias: [X; N],
    ) -> Builder<X, NNIN, Vector<X, N>, Linear<X, IN, N, PREV>> {
        let weights = Matrix::new(weights);
        let bias = Vector::new(bias);
        let components = Linear { prev: self.components, weights, bias };
        Builder { components, _out: PhantomData, ..self }
    }
}
