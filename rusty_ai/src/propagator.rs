/*
use crate::{
    neural_network::NeuralNetwork, optimizer::sgd::SGD_, traits, LossFunction, MeanSquaredError,
    NNTrainer,
};
use std::iter::{Map, Zip};

#[must_use = "`Prop` must be consumed to do work."]
pub struct Prop<NN, I> {
    nn: NN,
    iter: I,
}

impl<NN, I> Prop<NN, I> {
    pub(crate) fn new(nn: NN, iter: I) -> Self {
        Prop { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, I> Prop<&'a NeuralNetwork<IN, OUT>, I>
where I: Iterator<Item = &'a [f64; IN]>
{
    pub fn outputs(self) -> Map<I, impl FnMut(&'a [f64; IN]) -> [f64; OUT]> {
        self.iter.map(|input| self.nn._propagate(input))
    }

    pub fn outputs_losses(self, loss_function: impl LossFunction<OUT>) {
        self.outputs().map(|out| (out, loss_function.propagate_arr(&out)))
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Prop<&'a mut NNTrainer<IN, OUT, L, O>, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    I: Iterator<Item = &'a [f64; IN]>,
    EO: 'a,
{
    pub fn backpropagate(self, expected_outputs: impl Iterator<Item = &'a EO>) {
        self.iter.map(|input| {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagation(&out, expected_output);
        })
    }
}

pub trait Propagator<const IN: usize, const OUT: usize> {
    fn propagate_arr(&self, input: &[f64; IN]) -> [f64; OUT];

    /*
    fn propagate<'a, B>(&'a self, batch: B) -> Prop<Self, B::IntoIter>
    where
        Self: Sized,
        B: IntoIterator<Item = &'a [f64; IN]> + 'a,
    {
        Prop { nn: &self, iter: batch.into_iter() }
    }

    fn propagate_pairs<'a, B, EO>(
        &'a self,
        batch: B,
    ) -> PairProp<Self, Map<B::IntoIter, impl FnMut(B::Item) -> (&'a [f64; IN], &'a EO)>>
    where
        Self: Sized,
        B: IntoIterator + 'a,
        B::Item: Into<(&'a [f64; IN], &'a EO)>,
        EO: 'a,
    {
        PairProp { nn: &self, iter: batch.into_iter().map(Into::into) }
    }
    */
}

/*
#[must_use = "`Prop` must be consumed to do work."]
pub struct Prop<'a, NN, I> {
    nn: &'a NN,
    iter: I,
}

impl<'a, const IN: usize, NN, I> Prop<'a, NN, I>
where I: Iterator<Item = &'a [f64; IN]>
{
    pub fn expected_outputs<B, EO>(
        self,
        expected_outputs: B,
    ) -> PairProp<'a, NN, Zip<I, B::IntoIter>>
    where
        B: IntoIterator<Item = &'a EO>,
        EO: 'a,
    {
        PairProp { nn: self.nn, iter: self.iter.zip(expected_outputs) }
    }

    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn outputs<const OUT: usize>(self) -> Map<I, impl FnMut(I::Item) -> [f64; OUT] + 'a>
    where NN: Propagator<IN, OUT> {
        self.iter.map(|input| self.nn.propagate_arr(input))
    }
}

#[must_use = "`PairProp` must be consumed to do work."]
pub struct PairProp<'a, NN, I> {
    nn: &'a NN,
    iter: I,
}

impl<'a, const IN: usize, NN, I, EO> PairProp<'a, NN, I>
where
    I: Iterator<Item = (&'a [f64; IN], &'a EO)>,
    EO: 'a,
{
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn outputs<const OUT: usize>(self) -> Map<I, impl FnMut(I::Item) -> [f64; OUT] + 'a>
    where NN: Propagator<IN, OUT> {
        self.iter.map(|(input, _)| self.nn.propagate_arr(input))
    }
}

impl<'a, const IN: usize, const OUT: usize, I, EO> PairProp<'a, NeuralNetwork<IN, OUT>, I>
where
    I: Iterator<Item = (&'a [f64; IN], &'a EO)>,
    EO: 'a,
{
    pub fn outputs_losses(
        self,
        loss_function: &'a impl LossFunction<OUT, ExpectedOutput = EO>,
    ) -> Map<I, impl FnMut(I::Item) -> ([f64; OUT], f64) + 'a> {
        self.iter.map(|(input, eo)| {
            let out = self.nn.propagate_arr(input);
            let loss = loss_function.propagate_arr(&out, eo);
            (out, loss)
        })
    }

    pub fn losses(
        self,
        loss_function: &'a impl LossFunction<OUT, ExpectedOutput = EO>,
    ) -> Map<I, impl FnMut(I::Item) -> f64 + 'a> {
        self.iter.map(|(input, eo)| {
            let out = self.nn.propagate_arr(input);
            loss_function.propagate_arr(&out, eo)
        })
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I, EO> PairProp<'a, NNTrainer<IN, OUT, L, O>, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    I: Iterator<Item = (&'a [f64; IN], &'a EO)>,
    EO: 'a,
{
    pub fn outputs_losses(self) -> Map<I, impl FnMut(I::Item) -> ([f64; OUT], f64) + 'a> {
        self.iter.map(|(input, eo)| {
            let out = self.nn.propagate_arr(input);
            let loss = self.nn.get_loss_function().propagate_arr(&out, eo);
            (out, loss)
        })
    }

    pub fn losses(self) -> Map<I, impl FnMut(I::Item) -> f64 + 'a> {
        self.iter.map(|(input, eo)| {
            let out = self.nn.propagate_arr(input);
            self.nn.get_loss_function().propagate_arr(&out, eo)
        })
    }
}
*/
*/
