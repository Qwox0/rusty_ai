use crate::{prelude::*, gradient::Gradient};

pub trait Propagator<const IN: usize, const OUT: usize> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT>;

    fn propagate_many(&self, input_list: &Vec<[f64; IN]>) -> Vec<PropagationResult<OUT>>;

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT>;
}

pub trait Trainable<const IN: usize, const OUT: usize> {
    type Trainee;

    fn train(
        &mut self,
        training_data: &PairList<IN, OUT>,
        training_amount: usize,
        epoch_count: usize,
        callback: impl FnMut(usize, &Self::Trainee),
    );

    /// Trains the neural network for one generation/epoch. Uses a small data set `data_pairs` to
    /// find an aproximation for the weights gradient. The neural network's Optimizer changes the
    /// weights by using the calculated gradient.
    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>);

    fn optimize(&mut self, gradient: Gradient);
}
