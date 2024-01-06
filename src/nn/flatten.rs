use super::component::NNComponent;
use const_tensor::{Element, Len, Tensor, Vector};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Flatten<T, PREV> {
    pub(super) prev: PREV,
    pub(super) _tensor: PhantomData<T>,
}

impl<X, T, const LEN: usize, NNIN, PREV> NNComponent<X, NNIN, Vector<X, LEN>> for Flatten<T, PREV>
where
    X: Element,
    T: Tensor<X>,
    T::Data: Len<LEN>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    type Grad = PREV::Grad;
    type In = NNIN;
    type StoredData = PREV::StoredData;

    #[inline]
    fn prop(&self, input: NNIN) -> Vector<X, LEN> {
        let input = self.prev.prop(input);
        input.into_1d()
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (Vector<X, LEN>, Self::StoredData) {
        let (input, data) = self.prev.train_prop(input);
        (input.into_1d(), data)
    }

    #[inline]
    fn backprop(&self, data: PREV::StoredData, out_grad: Vector<X, LEN>, grad: &mut PREV::Grad) {
        let input_grad = T::from_1d(out_grad);
        self.prev.backprop(data, input_grad, grad)
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
