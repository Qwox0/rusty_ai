use super::component::NNComponent;
use const_tensor::{vector, Element, Len, Tensor, Vector};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Flatten<T, PREV> {
    pub(super) prev: PREV,
    _tensor: PhantomData<T>,
}

impl<X, T, const LEN: usize, NNIN, PREV> NNComponent<X, NNIN, Vector<X, LEN>> for Flatten<T, PREV>
where
    X: Element,
    T: Tensor<X>,
    T::Data: Len<LEN>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    #[inline]
    fn prop(&self, input: NNIN) -> Vector<X, LEN> {
        let input = self.prev.prop(input);
        input.into_1d()
    }

    #[inline]
    fn backprop(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use const_tensor::Tensor4;

    #[test]
    fn simple_flatten() {
        let tensor = Tensor4::new([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]);
        println!("tensor = {:?}", tensor);
        let flatten = Flatten { prev: (), _tensor: PhantomData };
        let vec = flatten.prop(tensor);
        println!("vec = {:?}", vec);
    }
}
