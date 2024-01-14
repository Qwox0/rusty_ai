//! # NN module

/*
use crate::trainer::{
    markers::{NoLossFunction, NoOptimizer},
    NNTrainerBuilder,
};
*/
use const_tensor::{Element, Shape, Tensor};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    iter::Map,
    marker::PhantomData,
};

pub mod builder;
mod component;
mod flatten;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

use self::component::NNDisplay;
pub use builder::NNBuilder;
pub use component::NNComponent;
pub use flatten::Flatten;
pub use linear::Linear;
pub use relu::{leaky_relu, relu, LeakyReLU, ReLU};
pub use sigmoid::{sigmoid, Sigmoid};
//pub use softmax::{LogSoftmax, Softmax};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NN<X: Element, IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT>> {
    components: C,
    _marker: PhantomData<(X, IN, OUT)>,
}

impl<X: Element, IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT>> NN<X, IN, OUT, C> {
    /// use [`NNBuilder`] instead!
    #[inline]
    fn new(components: C) -> NN<X, IN, OUT, C> {
        NN { components, _marker: PhantomData }
    }

    /*
    /// Converts `self` to a [`NNTrainerBuilder`] that can be used to create a [`NNTrainer`]
    ///
    /// Used to
    #[inline]
    pub fn to_trainer(self) -> NNTrainerBuilder<X, IN, OUT, NoLossFunction, NoOptimizer> {
        NNTrainerBuilder::new(self)
    }
    */
}

impl<X: Element, IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT>> NN<X, IN, OUT, C> {
    /*
    /// Creates a [`Gradient`] with the same dimensions as `self` and every element initialized to
    /// `0.0`
    pub fn init_zero_gradient(&self) -> Gradient<X> {
        self.layers.iter().map(Layer::init_zero_gradient).collect()
    }
    */

    /// Propagates a [`Tensor`] through the neural network and returns the output [`Tensor`].
    pub fn propagate(&self, input: &Tensor<X, IN>) -> Tensor<X, OUT> {
        // the compiler should inline all prop functions.
        self.components.prop(input.clone())
    }

    /// Iterates over a `batch` of inputs, propagates them and returns an [`Iterator`] over the
    /// outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn propagate_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> Tensor<X, OUT> + 'a>
    where
        B: IntoIterator<Item = &'a Tensor<X, IN>>,
    {
        batch.into_iter().map(|i| self.propagate(i))
    }

    /// Propagates a [`Tensor`] through the neural network and returns the output [`Tensor`] and
    /// additional data which is required for backpropagation.
    ///
    /// If only the output is needed, use the normal `propagate` method instead.
    pub fn training_propagate(&self, input: &Tensor<X, IN>) -> (Tensor<X, OUT>, C::StoredData) {
        self.components.train_prop(input.clone())
    }

    /// # Params
    ///
    /// `output_gradient`: gradient of the loss with respect to the network output.
    /// `train_data`: additional data create by `training_propagate` and required for
    /// backpropagation.
    /// `gradient`: stores the changes to each parameter. Has to have the same
    /// dimensions as `self`.
    pub fn backpropagate(
        &self,
        output_gradient: Tensor<X, OUT>,
        train_data: C::StoredData,
        gradient: &mut C::Grad,
    ) {
        self.components.backprop(output_gradient, train_data, gradient)
    }

    /*
    /// Tests the neural network.
    pub fn test<L: LossFunction<X, OUT>>(
        &self,
        input: IN,
        expected_output: &L::ExpectedOutput,
        loss_function: &L,
    ) -> ([X; OUT], X) {
        let out = self.propagate(input);
        let loss = loss_function.propagate(&out, expected_output);
        (out, loss)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// output losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to get the outputs use `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    pub fn test_batch<'a, B, L, EO: 'a>(
        &'a self,
        batch: B,
        loss_fn: &'a L,
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> ([X; OUT], X) + 'a>
    where
        B: IntoIterator<Item = (&'a Input<X, IN>, &'a EO)>,
        L: LossFunction<X, OUT, ExpectedOutput = EO>,
    {
        batch.into_iter().map(|(input, eo)| self.test(input, eo, loss_fn))
    }
    */
}

/*
impl<'a, X, const IN: usize, const OUT: usize> Default for NN<X, IN, OUT>
where
    X: Float,
    rand_distr::StandardNormal: rand_distr::Distribution<X>,
{
    /// creates a [`NN`] with one layer. Every parameter is equal to `0.0`.
    ///
    /// This is probably only useful for testing.
    fn default() -> Self {
        NNBuilder::default()
            .element_type()
            .input()
            .layer_from_parameters(Matrix::with_zeros(IN, OUT), vec![X::zero(); OUT].into())
            .build()
    }
}
*/

impl<X: Element, IN: Shape, OUT: Shape, C: NNComponent<X, IN, OUT>> Display for NN<X, IN, OUT, C>
where
    [(); IN::DIM]: Sized,
    [(); OUT::DIM]: Sized,
    for<'a> NNDisplay<'a, C>: fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NN: Tensor({:?}) -> Tensor({:?})", IN::get_dims_arr(), OUT::get_dims_arr())?;
        write!(f, "{}", NNDisplay(&self.components))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[test]
    fn display() {}

    #[test]
    fn serde() {
        #[derive(Debug, Serialize, Deserialize)]
        struct A {
            val: Box<[i32; 10]>,
        }

        let a = A { val: Box::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) };
        println!("{:?}", a);
        let json = serde_json::to_string(&a).unwrap();
        println!("{:?}", json);
        let a: A = serde_json::from_str(&json).unwrap();
        println!("{:?}", a);
        panic!()
    }
}
