use super::{Training, TrainingLosses, TrainingOutputs};
use crate::{
    loss_function::LossFunction, optimizer::Optimizer, prelude::Pair, trainer::NNTrainer, Input,
};
use matrix::{Float, Num};

impl<'a, X: Float, const IN: usize, const OUT: usize, L, EO, O>
    Training<'a, NNTrainer<X, IN, OUT, L, O>, Pair<X, IN, EO>>
where
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: Optimizer<X>,
    EO: 'a + Sync,
{
    /// single threaded version of `execute`.
    pub fn execute_single_thread(self) {
        self.nn.maybe_set_zero_gradient();
        for (input, eo) in self.data {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagate(&out, eo)
        }
        self.nn.clip_gradient();
        self.nn.optimize_trainee();
    }

    /// single threaded version of `outputs`.
    pub fn outputs_single_thread(
        self,
    ) -> TrainingOutputs<'a, X, IN, OUT, L, O, core::slice::Iter<'a, Pair<X, IN, EO>>> {
        TrainingOutputs::new(self.nn, self.data.iter())
    }

    /// single threaded version of `losses`.
    pub fn losses_single_thread(
        self,
    ) -> TrainingLosses<'a, X, IN, OUT, L, O, core::slice::Iter<'a, Pair<X, IN, EO>>> {
        TrainingLosses::new(self.nn, self.data.iter())
    }

    /// single threaded version of `mean_loss`.
    pub fn mean_loss_single_thread(self) -> X {
        let mut count = 0;
        let sum = self.losses_single_thread().fold(X::zero(), |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count.cast()
    }
}

impl<'a, X: Float, const IN: usize, const OUT: usize, L, O, I> Iterator
    for TrainingOutputs<'a, X, IN, OUT, L, O, I>
where
    L: LossFunction<X, OUT>,
    O: Optimizer<X>,
    I: ExactSizeIterator<Item = &'a (Input<X, IN>, L::ExpectedOutput)>,
{
    type Item = [X; OUT];

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

impl<'a, X: Float, const IN: usize, const OUT: usize, L, O, I> ExactSizeIterator
    for TrainingOutputs<'a, X, IN, OUT, L, O, I>
where
    L: LossFunction<X, OUT>,
    O: Optimizer<X>,
    I: ExactSizeIterator<Item = &'a (Input<X, IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

impl<'a, X: Float, const IN: usize, const OUT: usize, L, O, I> Iterator
    for TrainingLosses<'a, X, IN, OUT, L, O, I>
where
    L: LossFunction<X, OUT>,
    O: Optimizer<X>,
    I: ExactSizeIterator<Item = &'a (Input<X, IN>, L::ExpectedOutput)>,
{
    type Item = ([X; OUT], X);

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

impl<'a, X: Float, const IN: usize, const OUT: usize, L, O, I> ExactSizeIterator
    for TrainingLosses<'a, X, IN, OUT, L, O, I>
where
    L: LossFunction<X, OUT>,
    O: Optimizer<X>,
    I: ExactSizeIterator<Item = &'a (Input<X, IN>, L::ExpectedOutput)>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}
