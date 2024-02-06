use crate::{loss_function::LossFunction, optimizer::Optimizer};
#[allow(unused_imports)]
use const_tensor::Tensor;
use const_tensor::{Element, Shape};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, iter::Map};

/// A trait for the components of a neural network.
///
/// A component should contain a generic for the previous components. see [`ReLU`].
///
/// `NNIN`: input of the first component. should be a [`Tensor`].
/// `OUT`: out of this component. should be a [`Tensor`].
pub trait NNComponent<X: Element, NNIN: Shape, OUT: Shape>:
    Sized + fmt::Debug + fmt::Display + PartialEq + Serialize + Send + Sync + 'static
{
    /// Gradient component
    type Grad: GradComponent<X>;

    /// Shape of this components Input tensor.
    ///
    /// currently unused
    type In: Shape;

    /// not the best implementation but has to work for now.
    type OptState<O: Optimizer<X>>: Sized + Send + Sync + 'static;

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
        opt_state: &mut Self::OptState<O>,
    );

    /// Creates a gradient with the same dimensions as `self` and every element initialized to
    /// `0.0`.
    fn init_zero_grad(&self) -> Self::Grad;
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O>;

    fn iter_param(&self) -> impl Iterator<Item = &X>;

    /// Iterates over a `batch` of inputs, propagates them and returns an [`Iterator`] over the
    /// outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    fn prop_batch<'a, B, I>(
        &'a self,
        batch: B,
    ) -> std::iter::Map<B::IntoIter, impl FnMut(B::Item) -> Tensor<X, OUT> + 'a>
    where
        B: IntoIterator<Item = &'a I>,
        I: ToOwned<Owned = Tensor<X, NNIN>> + 'a,
    {
        batch.into_iter().map(|i: &I| self.prop(i.to_owned()))
    }

    /// Tests the neural network.
    fn test<L: LossFunction<X, OUT>>(
        &self,
        pair: &Pair<X, NNIN, impl Borrow<L::ExpectedOutput>>,
        loss_function: &L,
    ) -> TestResult<X, OUT> {
        let (input, expected_output) = pair.as_tuple();
        let out = self.prop(input.to_owned());
        let loss = loss_function.propagate(&out, expected_output.borrow());
        TestResult::new(out, loss)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// output losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to get the outputs use `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    fn test_batch<'a, B, L, EO>(
        &'a self,
        batch: B,
        loss_fn: &'a L,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> TestResult<X, OUT>>
    where
        B: IntoIterator<Item = &'a Pair<X, NNIN, EO>>,
        L: LossFunction<X, OUT>,
        EO: Borrow<L::ExpectedOutput> + 'a,
    {
        batch.into_iter().map(|p| self.test(p, loss_fn))
    }
}

mod t {
    use std::{
        borrow::{Borrow, Cow},
        collections::HashMap,
        ops::Deref,
    };

    #[derive(Debug, Clone)]
    struct Outer;

    struct Inner;

    impl Deref for Outer {
        type Target = Inner;

        fn deref(&self) -> &Self::Target {
            todo!()
        }
    }

    impl Borrow<Inner> for Outer {
        fn borrow(&self) -> &Inner {
            todo!()
        }
    }

    impl AsRef<Inner> for Outer {
        fn as_ref(&self) -> &Inner {
            todo!()
        }
    }

    impl ToOwned for Inner {
        type Owned = Outer;

        fn to_owned(&self) -> Self::Owned {
            todo!()
        }
    }

    fn my_fn_iter<'a, B, I>(_x: B)
    where
        B: IntoIterator<Item = &'a I>,
        I: Borrow<Inner> + 'a,
    {
    }

    fn my_fn_iter2<'a, B, I>(b: B)
    where
        B: IntoIterator<Item = &'a I>,
        I: ToOwned<Owned = Outer> + 'a,
    {
        let a = b.into_iter().map(|i| i.to_owned());
    }

    fn test() {
        my_fn_iter([&Inner]);
        my_fn_iter([&Outer]);
        my_fn_iter(&[Inner]);
        my_fn_iter(&[Outer]);

        my_fn_iter2([&Inner]);
        my_fn_iter2([&Outer]);
        my_fn_iter2(&[Inner]);
        my_fn_iter2(&[Outer]);
    }

    fn tsdfasdfasfd() {
        let a = vec![1, 2, 3];
        let b = vec![1.0, 2.0, 3.0];
        let zip = a.into_iter().zip(b).collect::<Vec<_>>();
    }
}

#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NNHead;

impl<X: Element, NNIN: Shape> NNComponent<X, NNIN, NNIN> for NNHead {
    type Grad = ();
    type In = NNIN;
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
    fn optimize<O: Optimizer<X>>(&mut self, _grad: &(), optimizer: &O, _opt_state: &mut ()) {}

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {}

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {}

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        None.into_iter()
    }
}

impl fmt::Display for NNHead {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

pub trait GradComponent<X: Element>: Sized + Send + Sync + 'static {
    fn set_zero(&mut self);
    fn add_mut(&mut self, other: impl Borrow<Self>);

    fn iter_param(&self) -> impl Iterator<Item = &X>;
    fn iter_param_mut(&mut self) -> impl Iterator<Item = &mut X>;

    fn add(mut self, other: impl Borrow<Self>) -> Self {
        self.add_mut(other);
        self
    }
}

pub trait TrainData: Sized + Send + Sync + 'static {}

pub struct Data<T, PREV: TrainData> {
    pub prev: PREV,
    pub data: T,
}

impl<T: Sized + Send + Sync + 'static, PREV: TrainData> TrainData for Data<T, PREV> {}

/*
impl<X: Element, NNIN: Shape> NNComponent<X, NNIN, NNIN> for () {
    type Grad = ();
    type OptState<O: Optimizer<X>> = ();
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

    #[inline]
    fn optimize<O: Optimizer<X>>(self, _grad: (), optimizer: &O, _opt_state: ()) -> ((), ()) {
        ((), ())
    }
}
*/

impl<X: Element> GradComponent<X> for () {
    #[inline]
    fn set_zero(&mut self) {}

    #[inline]
    fn add_mut(&mut self, other: impl Borrow<Self>) {}

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

/// Helper trait for components which aren't affected by training. (activation functions, ...)
pub trait NoTrainComponent<X: Element, IN, OUT, PREV> {
    type StoredData;

    fn get_prev(&self) -> &PREV;
    fn get_prev_mut(&mut self) -> &mut PREV;

    fn prop(&self, input: IN) -> OUT;

    fn train_prop(&self, input: IN) -> (OUT, Self::StoredData);

    fn backprop(&self, out_grad: OUT, data: Self::StoredData) -> IN;
}

/// Implements [`NNComponent`] for types implementing [`NoTrainComponent`].
#[macro_export]
macro_rules! derive_nn_component {
    ($ty:ty : $in:ident -> $out:ident) => {
        impl<X, NNIN, $in, $out, PREV> NNComponent<X, NNIN, $out> for $ty
        where
            Self: NoTrainComponent<X, $in, $out, PREV>,
            X: Element,
            PREV: NNComponent<X, NNIN, $in>,
        {
            type Grad = PREV::Grad;
            type OptState<O: Optimizer<X>> = PREV::OptState<O>;
            type StoredData = Data<Self::StoredData, PREV::StoredData>;

            #[inline]
            fn prop(&self, input: NNIN) -> $out {
                let input = self.get_prev().prop(input);
                NoTrainComponent::prop(&self, input)
            }

            #[inline]
            fn train_prop(&self, input: NNIN) -> ($out, Self::StoredData) {
                let (input, prev_data) = self.get_prev().train_prop(input);
                let (out, data) = self.train_prop(input);
                (out, Data { data, prev: prev_data })
            }

            #[inline]
            fn backprop(&self, out_grad: $out, data: Self::StoredData, grad: &mut PREV::Grad) {
                let Data { prev: prev_data, data } = data;
                let input_grad = self.backprop(out_grad, data);
                self.prev.backprop(input_grad, prev_data, grad)
            }

            #[inline]
            fn optimize<O: Optimizer<X>>(
                &mut self,
                grad: &PREV::Grad,
                optimizer: &O,
                opt_state: &mut PREV::OptState<O>,
            ) {
                self.get_prev_mut().optimize(grad, optimizer, opt_state);
            }

            #[inline]
            fn init_zero_grad(&self) -> PREV::Grad {
                self.get_prev().init_zero_grad()
            }

            #[inline]
            fn init_opt_state<O: Optimizer<X>>(&self) -> PREV::OptState<O> {
                self.get_prev().init_opt_state()
            }

            #[inline]
            fn iter_param(&self) -> impl Iterator<Item = &X> {
                self.get_prev().iter_param()
            }
        }
    };
}
use super::{Pair, TestResult};
pub use derive_nn_component;
