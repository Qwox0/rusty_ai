use super::component::{Data, NNComponent};
use const_tensor::{Len, Num, Tensor, TensorData};

#[inline]
pub fn relu<X: Num>(x: X) -> X {
    if x.is_positive() { x } else { X::ZERO }
}

#[inline]
pub fn leaky_relu<X: Num>(x: X, leak_rate: X) -> X {
    if x.is_positive() { x } else { leak_rate }
}

#[derive(Debug)]
pub struct ReLU<PREV> {
    pub(super) prev: PREV,
}

impl<X, T, NNIN, PREV> NNComponent<X, NNIN, T> for ReLU<PREV>
where
    X: Num,
    T: Tensor<X>,
    T::Data: Len<{ T::Data::LEN }>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    type Grad = PREV::Grad;
    type In = T;
    type StoredData = Data<T, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> T {
        let input = self.prev.prop(input);
        input.map_elem(relu)
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (T, Self::StoredData) {
        todo!()
    }

    #[inline]
    fn backprop(&self, data: Self::StoredData, out_grad: T, grad: &mut PREV::Grad) {
        todo!()
    }
}

#[derive(Debug)]
pub struct LeakyReLU<X, PREV> {
    pub(super) prev: PREV,
    pub(super) leak_rate: X,
}

impl<X, T, NNIN, PREV> NNComponent<X, NNIN, T> for LeakyReLU<X, PREV>
where
    X: Num,
    T: Tensor<X>,
    T::Data: Len<{ T::Data::LEN }>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    type Grad = PREV::Grad;
    type In = T;
    type StoredData = Data<T, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> T {
        let input = self.prev.prop(input);
        input.map_elem(|x| leaky_relu(x, self.leak_rate))
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (T, Self::StoredData) {
        todo!()
    }

    fn backprop(&self, data: Self::StoredData, out_grad: T, grad: &mut PREV::Grad) {
        todo!()
    }
}
