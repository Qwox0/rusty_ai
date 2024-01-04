use super::component::NNComponent;
use const_tensor::{Len, Num, Tensor, TensorData};

#[inline]
pub fn relu<X: Num>(x: X) -> X {
    if x.is_positive() { x } else { X::ZERO }
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
    #[inline]
    fn prop(&self, input: NNIN) -> T {
        let input = self.prev.prop(input);
        input.map_elem(relu)
    }

    #[inline]
    fn backprop(&mut self) {
        todo!()
    }
}
