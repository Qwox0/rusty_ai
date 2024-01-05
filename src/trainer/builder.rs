/*
use super::NNTrainer;
use crate::{
    clip_gradient_norm::ClipGradientNorm, loss_function::LossFunction, util, NeuralNetwork, Norm,
    OptimizerValues,
};
#[allow(unused_imports)]
use crate::{Gradient, Optimizer};
use markers::*;

/// Markers uses by [`NNTrainerBuilder`].
pub mod markers {
    #[allow(unused_imports)]
    use crate::{loss_function::LossFunction, optimizer::Optimizer};

    /// Marker for an undefined [`LossFunction`].
    pub struct NoLossFunction;

    /// Marker for an undefined [`Optimizer`].
    pub struct NoOptimizer;
}

/// A Builder used to create a [`NNTrainer`] from and for a [`NeuralNetwork`].
pub struct NNTrainerBuilder<X, const IN: usize, const OUT: usize, L, O> {
    network: NeuralNetwork<X, IN, OUT>,
    loss_function: L,
    optimizer: O,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm>,
    training_threads: usize,
}

impl<X, const IN: usize, const OUT: usize>
    NNTrainerBuilder<X, IN, OUT, NoLossFunction, NoOptimizer>
{
    /// # Defaults
    ///
    /// `retain_gradient`: `false`
    /// `clip_gradient_norm`: [`None`]
    /// `training_threads`: [`std::thread::available_parallelism`]
    pub fn new(network: NeuralNetwork<X, IN, OUT>) -> Self {
        NNTrainerBuilder {
            network,
            loss_function: NoLossFunction,
            retain_gradient: false,
            optimizer: NoOptimizer,
            clip_gradient_norm: None,
            training_threads: util::cpu_count(),
        }
    }
}

impl<X, const IN: usize, const OUT: usize, L, O> NNTrainerBuilder<X, IN, OUT, L, O> {
    /// Sets the [`LossFunction`] used by the [`NNTrainer`].
    pub fn loss_function<L2>(self, loss_function: L2) -> NNTrainerBuilder<X, IN, OUT, L2, O>
    where L2: LossFunction<X, OUT> {
        NNTrainerBuilder { loss_function, ..self }
    }

    /// Sets the [`Optimizer`] used by the [`NNTrainer`].
    pub fn optimizer<O2>(self, optimizer: O2) -> NNTrainerBuilder<X, IN, OUT, L, O2>
    where O2: OptimizerValues<X> {
        NNTrainerBuilder { optimizer, ..self }
    }

    /// Sets whether the [`NNTrainer`] keeps or resets the [`Gradient`] between training steps.
    pub fn retain_gradient(mut self, retain_gradient: bool) -> Self {
        self.retain_gradient = retain_gradient;
        self
    }

    /// Activates and sets a [`ClipGradientNorm`].
    pub fn clip_gradient_norm(mut self, clip_gradient_norm: ClipGradientNorm) -> Self {
        let _ = self.clip_gradient_norm.insert(clip_gradient_norm);
        self
    }

    /// Activates and sets a new [`ClipGradientNorm`] which is created from the parameters.
    pub fn new_clip_gradient_norm(self, max_norm: f32, norm_type: Norm) -> Self {
        let clip_grad_norm = ClipGradientNorm::new(norm_type, max_norm);
        self.clip_gradient_norm(clip_grad_norm)
    }

    /// Sets the number of cpu threads used for training. currently not working (TODO)
    pub fn training_threads(mut self, count: usize) -> Self {
        self.training_threads = count;
        self
    }
}

impl<X: Float, const IN: usize, const OUT: usize, EO, L, O> NNTrainerBuilder<X, IN, OUT, L, O>
where
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: OptimizerValues<X>,
{
    /// Consumes the Builder to create a new [`NNTrainer`].
    pub fn build(self) -> NNTrainer<X, IN, OUT, L, O::Optimizer> {
        let NNTrainerBuilder {
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
            training_threads: _,
        } = self;
        let optimizer = optimizer.init_with_layers(network.get_layers()).into();

        if None == clip_gradient_norm {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        NNTrainer::new(network, loss_function, optimizer, retain_gradient, clip_gradient_norm)
    }
}
*/
