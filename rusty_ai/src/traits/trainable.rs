use crate::prelude::*;

pub trait Trainee {
    fn init_zero_gradient(&self) -> Gradient;
}

pub trait Trainer<const IN: usize, const OUT: usize> {
    type Trainee: Trainee;

    fn get_trainee(&self) -> &Self::Trainee;

    fn get_gradient_mut(&mut self) -> &mut Gradient;

    fn set_zero_gradient(&mut self) {
        self.get_gradient_mut().set_zero()
    }

    fn calc_gradient<'a>(&mut self, batch: impl IntoIterator<Item = &'a Pair<IN, OUT>>);

    fn clip_gradient(&mut self, clip_gradient_norm: ClipGradientNorm) {
        clip_gradient_norm.clip_gradient_pytorch(self.get_gradient_mut());
        //clip_gradient_norm.clip_gradient_pytorch_device(&mut self.gradient);
    }

    fn optimize_trainee(&mut self);
    // ====

    /// Trains the neural network for one step/generation. Uses a small data set `data_pairs` to
    /// find an approximation for the weights gradient. The neural network's Optimizer changes the
    /// weights by using the calculated gradient.
    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>);

    /// Trains the `Self::Trainee` for `epoch_count` epochs. Each epoch the entire `training_data`
    /// is used to calculated the gradient approximation.
    fn full_train(
        &mut self,
        training_data: &PairList<IN, OUT>,
        epoch_count: usize,
        mut callback: impl FnMut(usize, &Self::Trainee),
    ) {
        for epoch in 1..=epoch_count {
            self.training_step(training_data.iter());
            callback(epoch, self.get_trainee());
        }
    }

    fn train(
        &mut self,
        training_data: &PairList<IN, OUT>,
        training_amount: usize,
        epoch_count: usize,
        mut callback: impl FnMut(usize, &Self::Trainee),
    ) {
        let mut rng = rand::thread_rng();
        for epoch in 1..=epoch_count {
            let training_data = training_data.choose_multiple(&mut rng, training_amount);
            self.training_step(training_data);
            callback(epoch, self.get_trainee());
        }
    }
}
