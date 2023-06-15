use crate::prelude::*;
use crate::util::{Norm, ScalarDiv};
use crate::{
    optimizer::IsOptimizer,
    traits::{Propagator, Trainable},
    util::{constructor, ScalarMul},
};

#[derive(Debug)]
pub struct ClipGradientNorm {
    pub norm_type: Norm,
    pub max_norm: f64,
}

#[derive(Debug)]
pub struct TrainableNeuralNetwork<const IN: usize, const OUT: usize> {
    network: NeuralNetwork<IN, OUT>,
    optimizer: Optimizer,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> TrainableNeuralNetwork<IN, OUT> {
    constructor! { pub(crate) new -> network: NeuralNetwork<IN, OUT>, optimizer: Optimizer, clip_gradient_norm: Option<ClipGradientNorm> }

    /// calculates the outputs and derivatives of all layers
    fn verbose_propagate(&self, input: &[f64; IN]) -> VerbosePropagation {
        let layer_count = self.network.get_layers().len();
        let mut outputs = Vec::with_capacity(layer_count + 1);
        outputs.push(input.to_vec());
        let mut derivatives = Vec::with_capacity(layer_count);

        for layer in self.network.iter_layers() {
            let input = outputs.last().expect("last element must exists");
            let (output, derivative) = layer.training_calculate(input);
            outputs.push(output);
            derivatives.push(derivative);
        }
        VerbosePropagation::new(outputs, derivatives)
    }
}

impl<const IN: usize, const OUT: usize> Trainable<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    type Trainee = NeuralNetwork<IN, OUT>;

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
            callback(epoch, &self.network);
        }
    }

    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>) {
        let mut data_count = 0;
        let mut gradient = self.network.init_zero_gradient();
        for (input, expected_output) in data_pairs.into_iter().map(Into::into) {
            self.network.backpropagation(
                self.verbose_propagate(input),
                expected_output,
                &mut gradient,
            );
            data_count += 1;
        }
        /*
        if data_count < 5 {
            eprintln!("WARN: Small training sets result in inaccurate gradients which might cause exploding weight values!")
        }
        */

        if let Some(clip_gradient_norm) = &self.clip_gradient_norm {
            let iter = gradient.iter_numbers().cloned();
            let norm = clip_gradient_norm.norm_type.calculate(iter);
            let clip_factor = clip_gradient_norm.max_norm / (norm + 1e-6); // 1e-6 copied from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
            gradient.mul_scalar_mut(clip_factor);
        }

        // average of all gradients
        gradient.div_scalar_mut(data_count as f64);

        self.optimize(gradient);
    }

    /*
    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a DataPair<IN, OUT>>) {
        let mut data_count = 0;
        let mut gradient = self.network.init_zero_gradient();
        for (input, expected_output) in data_pairs.into_iter().map(Into::into) {
            self.network.backpropagation(
                self.verbose_propagate(input),
                expected_output,
                &mut gradient,
            );
            data_count += 1;
        }
        /*
        if data_count < 5 {
            eprintln!("WARN: Small training sets result in inaccurate gradients which might cause exploding weight values!")
        }
        */

        // average of all gradients
        gradient.normalize(data_count);

        self.optimize(gradient);
    }
    */

    fn optimize(&mut self, gradient: Gradient) {
        self.optimizer.optimize_weights(&mut self.network, gradient);
        self.network.increment_generation();
    }
}

impl<const IN: usize, const OUT: usize> Propagator<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    fn propagate(&self, input: &[f64; IN]) -> crate::results::PropagationResult<OUT> {
        self.network.propagate(input)
    }

    fn propagate_many(
        &self,
        input_list: &Vec<[f64; IN]>,
    ) -> Vec<crate::results::PropagationResult<OUT>> {
        self.network.propagate_many(input_list)
    }

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> crate::results::TestsResult<OUT> {
        self.network.test_propagate(data_pairs)
    }
}

impl<const IN: usize, const OUT: usize> std::fmt::Display for TrainableNeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n; Optimizer: {:?}", self.network, self.optimizer)
    }
}

#[cfg(test)]
mod benches;
