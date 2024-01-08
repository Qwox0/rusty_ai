use const_tensor::{Element, Tensor};
use std::fmt::Debug;

pub trait NNComponent<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>>: Debug + Sized {
    /// Gradient component
    type Grad: GradComponent; //: NNComponent<X, NNIN, OUT>;
    /// Input tensor of the component
    ///
    /// currently unused
    type In: Tensor<X>;
    /// The data which is saved during `train_prop` and used in `backprop`.
    type StoredData: TrainData;

    /// Propagates the `input` [`Tensor`] through the entire sub network and then through this
    /// component.
    fn prop(&self, input: NNIN) -> OUT;

    /// Like `prop` but also returns the required data for backpropagation.
    fn train_prop(&self, input: NNIN) -> (OUT, Self::StoredData);

    /// Backpropagates the output gradient through this component and then backwards through the
    /// previous components.
    fn backprop(&self, out_grad: OUT, data: Self::StoredData, grad: &mut Self::Grad);
}

pub trait GradComponent {}

pub trait TrainData {}

pub struct Data<T, PREV: TrainData> {
    pub prev: PREV,
    pub data: T,
}

impl<T, PREV: TrainData> TrainData for Data<T, PREV> {}

// nn head

impl<X: Element, NNIN: Tensor<X>> NNComponent<X, NNIN, NNIN> for () {
    type Grad = ();
    type In = NNIN;
    type StoredData = ();

    #[inline]
    fn prop(&self, input: NNIN) -> NNIN {
        input
    }

    #[inline]
    fn train_prop(&self, input: NNIN) -> (NNIN, Self::StoredData) {
        (input, ())
    }

    #[inline]
    fn backprop(&self, _out_grad: NNIN, _data: (), _grad: &mut ()) {}
}

impl GradComponent for () {}

impl TrainData for () {}
