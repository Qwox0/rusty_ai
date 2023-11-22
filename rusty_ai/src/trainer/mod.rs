//! Trainer module.

use crate::{clip_gradient_norm::ClipGradientNorm, training::Training, *};
use loss_function::LossFunction;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, iter::Map};

mod builder;
pub use builder::{markers, NNTrainerBuilder};

/// Trainer for a [`NeuralNetwork`].
///
/// Can be constructed using a [`NNTrainerBuilder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NNTrainer<const IN: usize, const OUT: usize, L, O> {
    network: NeuralNetwork<IN, OUT>,
    gradient: Gradient,
    loss_function: L,
    retain_gradient: bool,
    optimizer: O,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize, L, O> NNTrainer<IN, OUT, L, O> {
    /// Returns a reference to the underlying [`NeuralNetwork`].
    pub fn get_network(&self) -> &NeuralNetwork<IN, OUT> {
        &self.network
    }

    /// Converts `self` into the underlying [`NeuralNetwork`]. This can be used after the training
    /// is finished.
    pub fn into_nn(self) -> NeuralNetwork<IN, OUT> {
        self.network
    }

    /// Propagates an [`Input`] through the underlying neural network and returns its output.
    #[inline]
    pub fn propagate(&self, input: &Input<IN>) -> [f64; OUT] {
        self.network.propagate(input)
    }

    /// Iterates over a `batch` of inputs and returns an [`Iterator`] over the outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn propagate_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(&'a Input<IN>) -> [f64; OUT]>
    where
        B: IntoIterator<Item = &'a Input<IN>>,
    {
        self.network.propagate_batch(batch)
    }

    /// Propagates an [`Input`] through the underlying neural network and returns the input and the
    /// outputs of every layer.
    ///
    /// If only the final output is needed, use `propagate` instead.
    ///
    /// This is used internally during training.
    #[inline]
    pub fn verbose_propagate(&self, input: &Input<IN>) -> VerbosePropagation<OUT> {
        self.network.verbose_propagate(input)
    }

    /// Sets every parameter of the interal [`Gradient`] to `0.0`.
    #[inline]
    pub fn set_zero_gradient(&mut self) {
        self.gradient.set_zero()
    }

    /// Sets every parameter of the interal [`Gradient`] to `0.0` if `self.retain_gradient` is
    /// `false`, otherwise this does nothing.
    ///
    /// If you always want to reset the [`Gradient`] use `set_zero_gradient` instead.
    pub fn maybe_set_zero_gradient(&mut self) {
        if !self.retain_gradient {
            self.set_zero_gradient();
        }
    }

    /// Clips the internal [`Gradient`] based on `self.clip_gradient_norm`.
    ///
    /// If `self.clip_gradient_norm` is [`None`], this does nothing.
    pub fn clip_gradient(&mut self) {
        if let Some(clip_gradient_norm) = self.clip_gradient_norm {
            clip_gradient_norm.clip_gradient_pytorch(&mut self.gradient);
            //clip_gradient_norm.clip_gradient_pytorch_device(&mut self.gradient);
        }
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> NNTrainer<IN, OUT, L, O>
where L: LossFunction<OUT, ExpectedOutput = EO>
{
    fn new(
        network: NeuralNetwork<IN, OUT>,
        loss_function: L,
        optimizer: O,
        retain_gradient: bool,
        clip_gradient_norm: Option<ClipGradientNorm>,
    ) -> Self {
        let gradient = network.init_zero_gradient();
        Self { network, gradient, loss_function, retain_gradient, optimizer, clip_gradient_norm }
    }

    /// Gets the [`LossFunction`] used during training.
    #[inline]
    pub fn get_loss_function(&self) -> &L {
        &self.loss_function
    }

    /// Propagate a [`VerbosePropagation`] Result backwards through the Neural
    /// Network. This modifies the internal [`Gradient`].
    ///
    /// # Math
    ///
    ///    L-1                   L
    /// o_(L-1)_0
    ///                      z_0 -> o_L_0
    /// o_(L-1)_1    w_ij                    C
    ///                      z_1 -> o_L_1
    /// o_(L-1)_2
    ///         j              i        i
    /// n_(L-1) = 3           n_L = 2
    ///
    /// L: current Layer with n_L Neurons called L_1, L_2, ..., L_n
    /// L-1: previous Layer with n_(L-1) Neurons
    /// o_L_i: output of Neuron L_i
    /// e_i: expected output of Neuron L_i
    /// Cost: C = 0.5 * ∑ (o_L_i - e_i)^2 from i = 1 to n_L
    /// -> dC/do_L_i = o_L_i - e_i
    ///
    /// f: activation function
    /// activation: o_L_i = f(z_i)
    /// -> do_L_i/dz_i = f'(z_i)
    ///
    /// -> dC/dz_i = dC/do_L_i * do_L_i/dz_i = (o_L_i - e_i) * f'(z_i)
    ///
    /// w_ij: weight of connection from (L-1)_j to L_i
    /// b_L: bias of Layer L
    /// weighted sum: z_i = b_L + ∑ w_ij * o_(L-1)_j from j = 1 to n_(L-1)
    /// -> dz_i/dw_ij      = o_(L-1)_j
    /// -> dz_i/do_(L-1)_j = w_ij
    /// -> dz_i/dw_ij      = 1
    ///
    ///
    /// dC/dw_ij      = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i) *  o_(L-1)_j
    /// dC/do_(L-1)_j = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i) *       w_ij
    /// dC/db_L       = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i)
    pub fn backpropagate(&mut self, verbose_prop: &VerbosePropagation<OUT>, expected_output: &EO) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let output_gradient = self.loss_function.backpropagate(verbose_prop, expected_output);

        self.network.backpropagate(verbose_prop, output_gradient, &mut self.gradient);
    }

    /// To test a batch of multiple pairs use `test_batch`.
    #[inline]
    pub fn test(&self, input: &Input<IN>, expected_output: &EO) -> ([f64; OUT], f64) {
        self.network.test(input, expected_output, &self.loss_function)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// outputs and the losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn test_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(&'a (Input<IN>, EO)) -> ([f64; OUT], f64)>
    where
        B: IntoIterator<Item = &'a (Input<IN>, EO)>,
        EO: 'a,
    {
        batch.into_iter().map(|(input, eo)| self.test(input, eo))
    }
}

impl<const IN: usize, const OUT: usize, L, O> NNTrainer<IN, OUT, L, O>
where O: Optimizer
{
    /// Uses the internal [`Optimizer`] and [`Gradient`] to optimize the internal
    /// [`NeuralNetwork`] once.
    #[inline]
    pub fn optimize_trainee(&mut self) {
        self.optimizer.optimize(&mut self.network, &self.gradient);
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
{
    /// Trains the internal [`NeuralNetwork`] lazily.
    #[inline]
    pub fn train<'a, B>(&'a mut self, batch: B) -> Training<Self, B::IntoIter>
    where
        B: IntoIterator<Item = &'a (Input<IN>, EO)>,
        EO: 'a,
    {
        Training::new(self, batch.into_iter())
    }

    /// creates a sample [`NNTrainer`].
    ///
    /// This is probably only useful for testing.
    pub fn default<V>() -> Self
    where
        L: Default,
        V: OptimizerValues<Optimizer = O> + Default,
    {
        NeuralNetwork::default()
            .to_trainer()
            .loss_function(L::default())
            .optimizer(V::default())
            .build()
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> Display for NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.network)?;
        write!(f, "Loss Function: {}, Optimizer: {}", self.loss_function, self.optimizer)
    }
}

#[cfg(test)]
mod benches;
