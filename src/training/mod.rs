/*
//! # Training iterator module

use crate::trainer::NNTrainer;
#[allow(unused_imports)]
use crate::NeuralNetwork;
use matrix::{Element, Num};

mod single_threaded;
mod with_rayon;

/// Represents a training step of a neural network.
///
/// The training is executed lazily, meaning this type must be consumed to perform calculations.
#[must_use = "`Training` must be consumed to do work."]
pub struct Training<'a, NN, P> {
    nn: &'a mut NN,
    data: &'a [P],
}

impl<'a, NN, P> Training<'a, NN, P> {
    #[inline]
    pub(crate) fn new(nn: &'a mut NN, data: &'a [P]) -> Self {
        Training { nn, data }
    }
}

/// [`Iterator`] over the outputs of a [`NeuralNetwork`] during training.
///
/// Created by `Training::outputs`.
#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingOutputs<'a, X: Element, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, X: Num, const IN: usize, const OUT: usize, L, O, I>
    TrainingOutputs<'a, X, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingOutputs { nn, iter }
    }
}

/// [`Iterator`] over the outputs and losses of a [`NeuralNetwork`] during training.
///
/// Created by `Training::losses`.
#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingLosses<'a, X, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, X: Num, const IN: usize, const OUT: usize, L, O, I>
    TrainingLosses<'a, X, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingLosses { nn, iter }
    }
}

// old

/*
//! # Training iterator module

#[allow(unused_imports)]
use crate::NeuralNetwork;
use crate::{input::Input, loss_function::LossFunction, optimizer::Optimizer, trainer::NNTrainer};

/// Represents a training step of a neural network.
///
/// The training is executed lazily, meaning this type must be consumed to perform calculations.
#[must_use = "`Training` must be consumed to do work."]
pub struct Training<'a, NN, I> {
    nn: &'a mut NN,
    iter: I,
}

impl<'a, NN, I> Training<'a, NN, I> {
    #[inline]
    pub(crate) fn new(nn: &'a mut NN, iter: I) -> Self {
        Training { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Training<'a, NNTrainer<X, IN, OUT, L, O>, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: ExactSizeIterator<Item = &'a (Input<IN>, EO)>,
    EO: 'a,
{
    /// Executes the [`Training`] and optimizes the [`NeuralNetwork`].
    pub fn execute(self) {
        self.nn.maybe_set_zero_gradient();
        for (input, eo) in self.iter {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagate(&out, eo)
        }
        self.nn.clip_gradient();
        self.nn.optimize_trainee();
    }

    /// Consumes `self` and returns an [`Iterator`] over the outputs of the [`NeuralNetwork`]
    /// calculated during the training.
    pub fn outputs(self) -> TrainingOutputs<'a, IN, OUT, L, O, I> {
        TrainingOutputs::new(self.nn, self.iter)
    }

    /// Consumes `self` and returns an [`Iterator`] over the outputs and losses of the
    /// [`NeuralNetwork`] calculated during the training.
    pub fn losses(self) -> TrainingLosses<'a, IN, OUT, L, O, I> {
        TrainingLosses::new(self.nn, self.iter)
    }

    /// Executes the [`Training`] and returns the mean of the losses calculated during training.
    pub fn mean_loss(self) -> f64 {
        let mut count = 0;
        let sum = self.losses().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as f64
    }
}

/// [`Iterator`] over the outputs of a [`NeuralNetwork`] during training.
///
/// Created by `Training::outputs`.
#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingOutputs<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I: ExactSizeIterator>
    TrainingOutputs<'a, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingOutputs { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> Iterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer,
    I: ExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    type Item = [f64; OUT];

    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);

        if self.iter.is_empty() {
            self.nn.clip_gradient();
            self.nn.optimize_trainee();
        }

        Some(out.get_nn_output())
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> ExactSizeIterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer,
    I: ExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

/// [`Iterator`] over the outputs and losses of a [`NeuralNetwork`] during training.
///
/// Created by `Training::losses`.
#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingLosses<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I: ExactSizeIterator>
    TrainingLosses<'a, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingLosses { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> Iterator
    for TrainingLosses<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer,
    I: ExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    type Item = ([f64; OUT], f64);

    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);
        let out = out.get_nn_output();
        let loss = self.nn.get_loss_function().propagate(&out, eo);

        if self.iter.is_empty() {
            self.nn.clip_gradient();
            self.nn.optimize_trainee();
        }

        Some((out, loss))
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> ExactSizeIterator
    for TrainingLosses<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer,
    I: ExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}
*/

// par: 1st try

/*
#[allow(unused_imports)]
use crate::NeuralNetwork;
use crate::{
    input::Input, loss_function::LossFunction, optimizer::Optimizer, trainer::NNTrainer, Gradient,
};
use rayon::iter::{IndexedParallelExactSizeIterator, ParallelExactSizeIterator};
use std::iter::Peekable;

/// Represents a training step of a neural network.
///
/// The training is executed lazily, meaning this type must be consumed to perform calculations.
#[must_use = "`Training` must be consumed to do work."]
pub struct Training<'a, NN, I> {
    nn: &'a mut NN,
    iter: I,
}

impl<'a, NN, I> Training<'a, NN, I> {
    #[inline]
    pub(crate) fn new(nn: &'a mut NN, iter: I) -> Self {
        Training { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Training<'a, NNTrainer<X, IN, OUT, L, O>, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO> + Send + Sync,
    O: Optimizer + Send + Sync,
    I: IndexedParallelExactSizeIterator<Item = &'a (Input<IN>, EO)>,
    EO: 'a,
{
    /// Executes the [`Training`] and optimizes the [`NeuralNetwork`].
    pub fn execute(self) {
        self.nn.maybe_set_zero_gradient();
        let gradient: Gradient = self
            .iter
            .map(|(input, eo)| {
                let out = self.nn.verbose_propagate(input);
                let output_gradient = self.nn.get_loss_function().backpropagate(&out, eo);

                let mut gradient = self.nn.get_network().init_zero_gradient();
                self.nn.get_network().backpropagate(&out, output_gradient, &mut gradient);
                gradient
            })
            .fold(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad)
            .reduce(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad);
        self.nn.unchecked_set_gradient(gradient);
        self.nn.optimize_trainee();
        self.nn.clip_gradient();
        self.nn.optimize_trainee();
    }

    /// Consumes `self` and returns an [`ExactSizeIterator`] over the outputs of the [`NeuralNetwork`]
    /// calculated during the training.
    pub fn outputs(self) -> TrainingOutputs<'a, IN, OUT, L, O, I> {
        self.nn.maybe_set_zero_gradient();
        //self.iter
        todo!()
    }

    /// Consumes `self` and returns an [`ExactSizeIterator`] over the outputs and losses of the
    /// [`NeuralNetwork`] calculated during the training.
    pub fn losses(self) -> TrainingLosses<'a, IN, OUT, L, O, I> {
        self.nn.maybe_set_zero_gradient();
        todo!()
    }

    /// Executes the [`Training`] and returns the mean of the losses calculated during training.
    pub fn mean_loss(self) -> f64 {
        todo!()
        /*
        let mut count = 0;
        let sum = self.losses().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as f64
        */
    }

    /*
    /// Consumes `self` and returns an [`ExactSizeIterator`] over the outputs of the [`NeuralNetwork`]
    /// calculated during the training.
    pub fn outputs(self) -> TrainingOutputs<'a, IN, OUT, L, O, I> {
        TrainingOutputs::new(self.nn, self.iter)
    }

    /// Consumes `self` and returns an [`ExactSizeIterator`] over the outputs and losses of the
    /// [`NeuralNetwork`] calculated during the training.
    pub fn losses(self) -> TrainingLosses<'a, IN, OUT, L, O, I> {
        TrainingLosses::new(self.nn, self.iter)
    }

    /// Executes the [`Training`] and returns the mean of the losses calculated during training.
    pub fn mean_loss(self) -> f64 {
        let mut count = 0;
        let sum = self.losses().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as f64
    }
    */
}

/// [`ExactSizeIterator`] over the outputs of a [`NeuralNetwork`] during training.
///
/// Created by `Training::outputs`.
#[must_use = "`ExactSizeIterators` must be consumed to do work."]
pub struct TrainingOutputs<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> TrainingOutputs<'a, IN, OUT, L, O, I> {
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingOutputs { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> ParallelExactSizeIterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT> + Send,
    O: Optimizer + Send,
    I: IndexedParallelExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    type Item = [f64; OUT];

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        todo!()
    }

    /*
    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);

        let is_last_iter = self.iter.peek().is_none();
        if is_last_iter {
            self.nn.clip_gradient();
            self.nn.optimize_trainee();
        }

        Some(out.get_nn_output())
    }
    */
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> IndexedParallelExactSizeIterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT> + Send,
    O: Optimizer + Send,
    I: IndexedParallelExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        todo!()
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        todo!()
    }
}

/// [`ExactSizeIterator`] over the outputs and losses of a [`NeuralNetwork`] during training.
///
/// Created by `Training::losses`.
#[must_use = "`ExactSizeIterators` must be consumed to do work."]
pub struct TrainingLosses<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<X, IN, OUT, L, O>,
    iter: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> TrainingLosses<'a, IN, OUT, L, O, I> {
    fn new(nn: &'a mut NNTrainer<X, IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingLosses { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> ParallelExactSizeIterator
    for TrainingLosses<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT> + Send,
    O: Optimizer + Send,
    I: ParallelExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    type Item = ([f64; OUT], f64);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        todo!()
    }

    /*
    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);
        let out = out.get_nn_output();
        let loss = self.nn.get_loss_function().propagate(&out, eo);

        let is_last_iter = self.iter.peek().is_none();
        if is_last_iter {
            self.nn.clip_gradient();
            self.nn.optimize_trainee();
        }

        Some((out, loss))
    }
    */
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> IndexedParallelExactSizeIterator
    for TrainingLosses<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT> + Send,
    O: Optimizer + Send,
    I: IndexedParallelExactSizeIterator<Item = &'a (Input<IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        todo!()
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        todo!()
    }
}
*/
*/
