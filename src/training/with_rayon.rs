use super::Training;
use crate::{loss_function::LossFunction, optimizer::Optimizer, prelude::Pair, trainer::NNTrainer};
use matrix::Float;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::mpsc;

impl<'a, X: Float, const IN: usize, const OUT: usize, L, EO, O>
    Training<'a, NNTrainer<X, IN, OUT, L, O>, Pair<X, IN, EO>>
where
    L: LossFunction<X, OUT, ExpectedOutput = EO> + Send + Sync,
    O: Optimizer<X> + Send + Sync,
    EO: 'a + Sync,
{
    /// Executes the [`Training`] and optimizes the [`NeuralNetwork`].
    pub fn execute(self) {
        self.nn.maybe_set_zero_gradient();
        let gradient = self
            .data
            .par_iter()
            .map(|(input, eo)| {
                let out = self.nn.verbose_propagate(input);
                let mut gradient = self.nn.get_network().init_zero_gradient();
                self.nn.backpropagate_into(&out, eo, &mut gradient);
                gradient
            })
            .fold(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad)
            .reduce(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad);
        self.nn.unchecked_add_gradient(gradient);
        self.nn.clip_gradient();
        self.nn.optimize_trainee();
    }

    /// Consumes `self` and returns an [`Iterator`] over the outputs of the [`NeuralNetwork`]
    /// calculated during the training.
    ///
    /// The order of the items isn't guaranteed!
    pub fn outputs(self) -> mpsc::IntoIter<[X; OUT]> {
        self.nn.maybe_set_zero_gradient();

        let (sender, receiver) = mpsc::channel();
        let gradient = self
            .data
            .par_iter()
            .map(|(input, eo)| {
                let out = self.nn.verbose_propagate(input);
                let mut gradient = self.nn.get_network().init_zero_gradient();
                self.nn.backpropagate_into(&out, eo, &mut gradient);
                sender.send(out.get_nn_output()).expect("could send output");
                gradient
            })
            .fold(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad)
            .reduce(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad);
        self.nn.unchecked_set_gradient(gradient);
        self.nn.clip_gradient();
        self.nn.optimize_trainee();

        receiver.into_iter()
    }

    /// Consumes `self` and returns an [`Iterator`] over the outputs and losses of the
    /// [`NeuralNetwork`] calculated during the training.
    ///
    /// The order of the items isn't guaranteed!
    pub fn losses(self) -> mpsc::IntoIter<([X; OUT], X)> {
        self.nn.maybe_set_zero_gradient();

        let (sender, receiver) = mpsc::channel();
        let gradient = self
            .data
            .par_iter()
            .map(|(input, eo)| {
                let out = self.nn.verbose_propagate(input);
                let mut gradient = self.nn.get_network().init_zero_gradient();
                self.nn.backpropagate_into(&out, eo, &mut gradient);

                let out = out.get_nn_output();
                let loss = self.nn.get_loss_function().propagate(&out, eo);
                sender.send((out, loss)).expect("could send output and loss");

                gradient
            })
            .fold(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad)
            .reduce(|| self.nn.get_network().init_zero_gradient(), |acc, grad| acc + grad);
        self.nn.unchecked_set_gradient(gradient);
        self.nn.clip_gradient();
        self.nn.optimize_trainee();

        receiver.into_iter()
    }

    /// Executes the [`Training`] and returns the mean of the losses calculated during training.
    pub fn mean_loss(self) -> X {
        todo!("mean_loss")
        /*
        let mut count = 0;
        let sum = self.losses().fold(0.0, |acc, (_, loss)| {
            count += 1;
            acc + loss
        });
        sum / count as X
        */
    }
}

/*
impl<'a, const IN: usize, const OUT: usize, L, O, I> Iterator
    for TrainingOutputs<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer<X>,
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
    O: Optimizer<X>,
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
    O: Optimizer<X>,
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
    O: Optimizer<X>,
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
