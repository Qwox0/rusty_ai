use crate::{
    optimizer::Optimizer,
    prelude::{LossFunction, NNTrainer},
};
use std::iter::Peekable;

#[must_use = "`Training` must be consumed to do work."]
pub struct Training<'a, NN, I> {
    nn: &'a mut NN,
    iter: I,
}

impl<'a, NN, I> Training<'a, NN, I> {
    pub(crate) fn new(nn: &'a mut NN, iter: I) -> Self {
        Training { nn, iter }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Training<'a, NNTrainer<IN, OUT, L, O>, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: Iterator<Item = &'a ([f64; IN], EO)>,
    EO: 'a,
{
    pub fn execute(self) {
        self.nn.maybe_set_zero_gradient();
        for (input, eo) in self.iter {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagate(&out, eo)
        }
        self.nn.maybe_clip_gradient();
        self.nn.optimize_trainee();
    }

    pub fn outputs(self) -> TrainingOutputs<'a, IN, OUT, L, O, I> {
        TrainingOutputs::new(self.nn, self.iter)
    }

    pub fn losses(self) -> TrainingLosses<'a, IN, OUT, L, O, I> {
        TrainingLosses::new(self.nn, self.iter)
    }

    pub fn mean_loss(self) -> f64 {
        let mut count = 0;
        let sum = self.losses().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as f64
    }
}

#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingOutputs<'a, const IN: usize, const OUT: usize, L, O, I: Iterator> {
    nn: &'a mut NNTrainer<IN, OUT, L, O>,
    iter: Peekable<I>,
    while_grad: bool,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I: Iterator>
    TrainingOutputs<'a, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingOutputs { nn, iter: iter.peekable(), while_grad: false }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Iterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: Iterator<Item = &'a ([f64; IN], EO)>,
    EO: 'a,
{
    type Item = [f64; OUT];

    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);

        let is_last_iter = self.iter.peek().is_none();
        if is_last_iter {
            self.nn.maybe_clip_gradient();
            self.nn.optimize_trainee();
        }

        Some(out.get_nn_output())
    }
}

#[must_use = "`Iterators` must be consumed to do work."]
pub struct TrainingLosses<'a, const IN: usize, const OUT: usize, L, O, I: Iterator> {
    nn: &'a mut NNTrainer<IN, OUT, L, O>,
    iter: Peekable<I>,
    while_grad: bool,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I: Iterator>
    TrainingLosses<'a, IN, OUT, L, O, I>
{
    fn new(nn: &'a mut NNTrainer<IN, OUT, L, O>, iter: I) -> Self {
        nn.maybe_set_zero_gradient();
        TrainingLosses { nn, iter: iter.peekable(), while_grad: false }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, EO, O, I> Iterator
    for TrainingLosses<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: Iterator<Item = &'a ([f64; IN], EO)>,
    EO: 'a,
{
    type Item = ([f64; OUT], f64);

    fn next(&mut self) -> Option<Self::Item> {
        let (input, eo) = self.iter.next()?;
        let out = self.nn.verbose_propagate(input);
        self.nn.backpropagate(&out, eo);
        let out = out.get_nn_output();
        let loss = self.nn.get_loss_function().propagate(&out, eo);

        let is_last_iter = self.iter.peek().is_none();
        if is_last_iter {
            self.nn.maybe_clip_gradient();
            self.nn.optimize_trainee();
        }

        Some((out, loss))
    }
}
