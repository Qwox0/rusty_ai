use super::NN;
use crate::optimizer::Optimizer;
use const_tensor::{Element, Len, Multidimensional, MultidimensionalOwned, Shape, Tensor, Vector};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Flatten<S: Shape, PREV> {
    pub(super) prev: PREV,
    #[serde(skip)]
    pub(super) _shape: PhantomData<S>,
}

impl<S: Shape, PREV> Flatten<S, PREV> {
    pub fn new(prev: PREV) -> Flatten<S, PREV> {
        Flatten { prev, _shape: PhantomData }
    }
}

impl<X, S, const LEN: usize, NNIN, PREV> NN<X, NNIN, [(); LEN]> for Flatten<S, PREV>
where
    X: Element,
    S: Shape + Len<LEN> + PartialEq,
    [(); S::DIM]: Sized,
    NNIN: Shape,
    PREV: NN<X, NNIN, S>,
{
    type Grad = PREV::Grad;
    type In = S;
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
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
    fn backprop_inplace(&self, out_grad: Vector<X, LEN>, data: PREV::StoredData, grad: &mut PREV::Grad) {
        let input_grad = Tensor::from_1d(out_grad);
        self.prev.backprop_inplace(input_grad, data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    ) {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev.iter_param()
    }
}

impl<'a, S: Shape, PREV> fmt::Display for Flatten<S, PREV>
where
    [(); S::DIM]: Sized,
    PREV: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        writeln!(f, "Flatten: Tensor({:?}) -> Vector({})", S::get_dims_arr(), S::LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::NNHead;

    #[test]
    fn simple_flatten() {
        use const_tensor::Tensor4;
        let tensor = Tensor4::new([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]);
        println!("tensor = {:?}", tensor);
        let flatten = Flatten::new(NNHead);
        let vec = flatten.prop(tensor);
        println!("vec = {:?}", vec);
    }
}
