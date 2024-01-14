/*
use super::component::{Data, NNComponent};
use const_tensor::{Len, Num, Tensor, TensorData};

#[derive(Debug)]
pub struct Softmax<PREV> {
    pub(super) prev: PREV,
}

impl<X, T, NNIN, PREV> NNComponent<X, NNIN, T> for Softmax<PREV>
where
    X: Num,
    T: Tensor<X>,
    T::Data: Len<{ T::Data::LEN }>,
    <T::Data as TensorData<X>>::Mapped<bool>: Len<{ T::Data::LEN }>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    type Grad = PREV::Grad;
    type In = T;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<T::Mapped<bool>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> T {
        let input = self.prev.prop(input);
        input.map_elem(relu)
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (T, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_elem(relu);
        let data = out.map_clone(|x| !x.is_zero());
        (out, Data { data, prev: prev_data })
    }

    /// ```
    /// o_i = relu(a_i)
    /// do_i/da_i = relu'(a_i)
    /// dL/da_i   = dL/do_i * relu'(a_i)
    ///
    /// -----
    ///
    /// o_i: output component i
    /// a_i: input component i
    ///
    /// L: total loss
    /// ```
    #[inline]
    fn backprop(&self, out_grad: T, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= X::from_bool(is_pos);
        }
        self.prev.backprop(input_grad, prev_data, grad)
    }
}

#[derive(Debug)]
pub struct LogSoftmax<X, PREV> {
    pub(super) prev: PREV,
    pub(super) leak_rate: X,
}

impl<X, T, NNIN, PREV> NNComponent<X, NNIN, T> for LogSoftmax<X, PREV>
where
    X: Num,
    T: Tensor<X>,
    T::Data: Len<{ T::Data::LEN }>,
    <T::Data as TensorData<X>>::Mapped<bool>: Len<{ T::Data::LEN }>,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, T>,
{
    type Grad = PREV::Grad;
    type In = T;
    /// The data which is saved during `train_prop` and used in `backprop`.
    ///
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<T::Mapped<bool>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: NNIN) -> T {
        let input = self.prev.prop(input);
        input.map_elem(|x| leaky_relu(x, self.leak_rate))
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (T, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let out = input.map_elem(|x| leaky_relu(x, self.leak_rate));
        let data = out.map_clone(|x| !x.is_zero());
        (out, Data { data, prev: prev_data })
    }

    #[inline]
    fn backprop(&self, out_grad: T, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= if is_pos { X::ONE } else { self.leak_rate }
        }
        self.prev.backprop(input_grad, prev_data, grad)
    }
}
*/
