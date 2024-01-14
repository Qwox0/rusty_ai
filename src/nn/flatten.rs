use super::component::NNComponent;
use crate::nn::component::NNDisplay;
use const_tensor::{Element, Len, Shape, Tensor, Vector};
use core::fmt;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Flatten<S, PREV> {
    pub(super) prev: PREV,
    pub(super) _shape: PhantomData<S>,
}

impl<S: Shape> Flatten<S, ()> {
    pub fn new() -> Flatten<S, ()> {
        Self::with_prev(())
    }
}

impl<S: Shape, PREV> Flatten<S, PREV> {
    pub fn with_prev(prev: PREV) -> Flatten<S, PREV> {
        Flatten { prev, _shape: PhantomData }
    }
}

impl<X, S, const LEN: usize, NNIN, PREV> NNComponent<X, NNIN, [(); LEN]> for Flatten<S, PREV>
where
    X: Element,
    S: Shape + Len<LEN>,
    [(); S::DIM]: Sized,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type StoredData = PREV::StoredData;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Vector<X, LEN> {
        let input = self.prev.prop(input);
        input.into_1d()
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Vector<X, LEN>, Self::StoredData) {
        let (input, data) = self.prev.train_prop(input);
        (input.into_1d(), data)
    }

    #[inline]
    fn backprop(&self, out_grad: Vector<X, LEN>, data: PREV::StoredData, grad: &mut PREV::Grad) {
        let input_grad = Tensor::from_1d(out_grad);
        self.prev.backprop(input_grad, data, grad)
    }
}

impl<'a, S: Shape, PREV> fmt::Display for NNDisplay<'a, Flatten<S, PREV>>
where
    [(); S::DIM]: Sized,
    NNDisplay<'a, PREV>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", NNDisplay(&self.0.prev))?;
        writeln!(f, "Flatten: Tensor({:?}) -> Vector({})", S::get_dims_arr(), S::LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_flatten() {
        use const_tensor::Tensor4;
        let tensor = Tensor4::new([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]);
        println!("tensor = {:?}", tensor);
        let flatten = Flatten::new();
        let vec = flatten.prop(tensor);
        println!("vec = {:?}", vec);
    }
}
