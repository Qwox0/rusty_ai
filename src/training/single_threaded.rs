use super::{Training, TrainingLosses, TrainingOutputs};
use crate::{
    loss_function::LossFunction, optimizer::Optimizer, prelude::Pair, trainer::NNTrainer, Input,
};

impl<'a, const IN: usize, const OUT: usize, L, EO, O>
    Training<'a, NNTrainer<IN, OUT, L, O>, Pair<IN, EO>>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
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
    ) -> TrainingOutputs<'a, IN, OUT, L, O, core::slice::Iter<'a, Pair<IN, EO>>> {
        TrainingOutputs::new(self.nn, self.data.iter())
    }

    /// single threaded version of `losses`.
    pub fn losses_single_thread(
        self,
    ) -> TrainingLosses<'a, IN, OUT, L, O, core::slice::Iter<'a, Pair<IN, EO>>> {
        TrainingLosses::new(self.nn, self.data.iter())
    }

    /// single threaded version of `mean_loss`.
    pub fn mean_loss_single_thread(self) -> f64 {
        let mut count = 0;
        let sum = self.losses_single_thread().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as f64
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

