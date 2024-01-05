//! # NN module

/*
use crate::trainer::{
    markers::{NoLossFunction, NoOptimizer},
    NNTrainerBuilder,
};
*/
use const_tensor::{Element, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

pub mod builder;
mod component;
mod flatten;
mod linear;
mod relu;

pub use builder::NNBuilder;
pub use component::NNComponent;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NN<X: Element, IN: Tensor<X>, OUT: Tensor<X>, C: NNComponent<X, IN, OUT>> {
    components: C,
    _marker: PhantomData<(X, IN, OUT)>,
}

impl<X: Element, IN: Tensor<X>, OUT: Tensor<X>, C: NNComponent<X, IN, OUT>> NN<X, IN, OUT, C> {
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

/*
impl<X: Float, const IN: usize, const OUT: usize> NN<X, IN, OUT> {
    /// Creates a [`Gradient`] with the same dimensions as `self` and every element initialized to
    /// `0.0`
    pub fn init_zero_gradient(&self) -> Gradient<X> {
        self.layers.iter().map(Layer::init_zero_gradient).collect()
    }

    /// Propagates an [`Input`] through the neural network and returns its output.
    pub fn propagate(&self, input: &Input<X, IN>) -> [X; OUT] {
        let input = Cow::from(input.as_slice());
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.propagate(&acc).into())
            .into_owned()
            .try_into()
            .expect("last layer should have `OUT` neurons")
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
    ) -> Map<B::IntoIter, impl FnMut(B::Item) -> [X; OUT] + 'a>
    where
        B: IntoIterator<Item = &'a Input<X, IN>>,
    {
        batch.into_iter().map(|i| self.propagate(i))
    }

    /// Propagates an [`Input`] through the neural network and returns the input and the outputs of
    /// every layer.
    ///
    /// If only the final output is needed, use `propagate` instead.
    ///
    /// This is used internally during training.
    pub fn verbose_propagate(&self, input: &Input<X, IN>) -> VerbosePropagation<X, OUT> {
        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        let nn_out = self.layers.iter().fold(input.to_vec(), |input, layer| {
            let out = layer.propagate(&input);
            outputs.push(input);
            out
        });
        outputs.push(nn_out);
        VerbosePropagation::new(outputs)
    }

    /// # Params
    ///
    /// `verbose_prop`: input and activation of every layer
    /// `nn_output_gradient`: gradient of the loss with respect to the network output (input to
    /// loss function)
    /// `gradient`: stores the changes to each parameter. Has to have the same dimensions as
    /// `self`.
    pub fn backpropagate(
        &self,
        verbose_prop: &VerbosePropagation<X, OUT>,
        nn_output_gradient: OutputGradient<X>,
        gradient: &mut Gradient<X>,
    ) {
        self.layers
            .iter()
            .zip(&mut gradient.layers)
            .zip(verbose_prop.iter_layers())
            .rev()
            .fold(nn_output_gradient, |output_gradient, ((layer, gradient), [input, output])| {
                layer.backpropagate(input, output, output_gradient, gradient)
            });
    }

    /// Tests the neural network.
    pub fn test<L: LossFunction<X, OUT>>(
        &self,
        input: &Input<X, IN>,
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
}

impl<X: Element, const IN: usize, const OUT: usize> ParamsIter<X> for NN<X, IN, OUT> {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a X> {
        self.layers.iter().map(Layer::iter).flatten()
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut X> {
        self.layers.iter_mut().map(Layer::iter_mut).flatten()
    }
}

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

impl<X: Display, const IN: usize, const OUT: usize> Display for NN<X, IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let get_plural_s = |x: usize| if x == 1 { "" } else { "s" };
        writeln!(
            f,
            "Neural Network: {IN} Input{} -> {OUT} Output{}",
            get_plural_s(IN),
            get_plural_s(OUT),
        )?;
        let layers_text =
            self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", layers_text)
    }
}
*/
