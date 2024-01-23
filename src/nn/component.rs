use crate::optimizer::Optimizer;
use const_tensor::{Element, Shape, Tensor};
use core::fmt;
use std::iter::Sum;

/// A trait for the components of a neural network.
pub trait NNComponent<X: Element, NNIN: Shape, OUT: Shape>:
    Sized + fmt::Debug + Send + Sync + 'static
{
    /// Gradient component
    type Grad: GradComponent<X>;

    // /// Shape of this components Input tensor.
    // type In: Shape;

    /// not the best implementation but has to work for now.
    type OptState<O: Optimizer<X>>;

    /// The data which is saved during `train_prop` and used in `backprop`.
    type StoredData: TrainData;

    /// Propagates the `input` [`Tensor`] through the entire sub network and then through this
    /// component.
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, OUT>;

    /// Like `prop` but also returns the required data for backpropagation.
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, OUT>, Self::StoredData);

    /// Backpropagates the output gradient through this component and then backwards through the
    /// previous components.
    fn backprop(&self, out_grad: Tensor<X, OUT>, data: Self::StoredData, grad: &mut Self::Grad);

    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: Self::OptState<O>,
    ) -> Self::OptState<O>;

    fn init_zero_grad(&self) -> Self::Grad;
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O>;

    fn iter_param(&self) -> impl Iterator<Item = &X>;
}

pub trait GradComponent<X: Element>: Sized {
    fn set_zero(&mut self);
    fn add_mut(&mut self, other: &Self);

    fn iter_param(&self) -> impl Iterator<Item = &X>;
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X>;

    fn add(mut self, other: &Self) -> Self {
        self.add_mut(other);
        self
    }
}

pub trait TrainData {}

pub struct Data<T, PREV: TrainData> {
    pub prev: PREV,
    pub data: T,
}

impl<T, PREV: TrainData> TrainData for Data<T, PREV> {}

/// Wrapper for [`NNComponent`] to implement [`fmt::Display`].
pub struct NNDisplay<'a, C>(pub &'a C);

/*
impl<X: Element, NNIN: Shape> NNComponent<X, NNIN, NNIN> for () {
    type Grad = ();
    type OptState<O: Optimizer<X>> = ();
    type StoredData = ();

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, NNIN> {
        input
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, NNIN>, Self::StoredData) {
        (input, ())
    }

    #[inline]
    fn backprop(&self, _out_grad: Tensor<X, NNIN>, _data: (), _grad: &mut ()) {}

    #[inline]
    fn optimize<O: Optimizer<X>>(self, _grad: (), optimizer: &O, _opt_state: ()) -> ((), ()) {
        ((), ())
    }
}
*/

impl<X: Element, NNIN: Shape> NNComponent<X, NNIN, NNIN> for () {
    type Grad = ();
    type OptState<O: Optimizer<X>> = ();
    type StoredData = ();

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, NNIN> {
        input
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, NNIN>, Self::StoredData) {
        (input, ())
    }

    #[inline]
    fn backprop(&self, _out_grad: Tensor<X, NNIN>, _data: (), _grad: &mut ()) {}

    #[inline]
    fn optimize<O: Optimizer<X>>(&mut self, _grad: &(), optimizer: &O, _opt_state: ()) -> () {}

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {}

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {}

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        None.into_iter()
    }
}

impl fmt::Display for NNDisplay<'_, ()> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<X: Element> GradComponent<X> for () {
    #[inline]
    fn set_zero(&mut self) {}

    #[inline]
    fn add_mut(&mut self, other: &Self) {}

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        None.into_iter()
    }

    #[inline]
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X> {
        None.into_iter()
    }
}

impl TrainData for () {}

macro_rules! component_new {
    ($ty:ident) => {
        component_new! { $ty<PREV> -> }
    };
    ($ty:ident < $( $gen:ident),+ > -> $( $param:ident : $paramty:ty ),* ) => {
        impl<$($gen),+> $ty<$($gen),+> {
            pub fn new( $($param : $paramty ,)* prev: PREV) -> Self {
                $ty { $($param , )* prev }
            }
        }
    };
}
pub(crate) use component_new;

/*
/// A component which is not affected by training. (activation functions, ...)
pub trait NoTrainComponent<X: Element, IN: Shape, OUT: Shape, PREV> {
    type StoredData;

    fn get_prev(&self) -> &PREV;
    fn get_prev_mut(&mut self) -> &mut PREV;

    fn prop(&self, input: Tensor<X, IN>) -> Tensor<X, OUT>;

    fn train_prop(&self, input: Tensor<X, IN>) -> (Tensor<X, OUT>, Self::StoredData);

    fn backprop(&self, out_grad: Tensor<X, OUT>, data: Self::StoredData) -> Tensor<X, IN>;
}

impl<X, NNIN, IN, OUT, C, PREV> NNComponent<X, NNIN, OUT> for C
where
    X: Element,
    NNIN: Shape,
    IN: Shape,
    OUT: Shape,
    C: NoTrainComponent<X, IN, OUT, PREV>,
    PREV: NNComponent<X, NNIN, IN>,
{
    type Grad = PREV::Grad;
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    type StoredData = Data<Self::StoredData, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, OUT> {
        NoTrainComponent::prop(&self, input)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Tensor<X, OUT>, Self::StoredData) {
        let (input, prev_data) = self.get_prev().train_prop(input);
        let (out, data) = self.train_prop(input);
        (out, Data { data, prev: prev_data })
    }

    #[inline]
    fn backprop(&self, out_grad: Tensor<X, OUT>, data: Self::StoredData, grad: &mut PREV::Grad) {
        let Data { prev: prev_data, data } = data;
        let input_grad = self.backprop(out_grad, data);
        self.prev.backprop(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        mut self,
        grad: Self::Grad,
        optimizer: &O,
        mut opt_state: Self::OptState<O>,
    ) -> (Self, Self::OptState<O>) {
        let prev = self.get_prev_mut();
        (*prev, opt_state) = prev.optimize(grad, optimizer, opt_state);
        (self, opt_state)
    }
}
*/
