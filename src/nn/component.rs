use const_tensor::{Element, Vector};
use std::fmt::Debug;

pub trait NNComponent<X: Element, const NNIN: usize, const OUT: usize>: Debug + Sized {
    fn prop(&self, input: Vector<X, NNIN>) -> Vector<X, OUT>;

    fn backprop(&mut self) {}

    fn layer<const N: usize>(self, weights: [[f32; OUT]; N]) -> Layer<OUT, N, Self> {
        Layer { item: self, weights: weights.into() }
    }

    fn relu(self) -> ReLU<Self> {
        ReLU { item: self }
    }

    fn build(self) -> NN<NNIN, OUT, Self> {
        NN(self)
    }
}
