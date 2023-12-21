use super::NNTrainer;
use crate::{
    clip_gradient_norm::ClipGradientNorm, loss_function::LossFunction, NeuralNetwork, Norm,
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
pub struct NNTrainerBuilder<const IN: usize, const OUT: usize, L, O> {
    network: NeuralNetwork<IN, OUT>,
    loss_function: L,
    optimizer: O,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
    /// # Defaults
    ///
    /// `retain_gradient`: `false`
    /// `clip_gradient_norm`: [`None`]
    pub fn new(network: NeuralNetwork<IN, OUT>) -> Self {
        NNTrainerBuilder {
            network,
            loss_function: NoLossFunction,
            retain_gradient: false,
            optimizer: NoOptimizer,
            clip_gradient_norm: None,
        }
    }
}

impl<const IN: usize, const OUT: usize, L, O> NNTrainerBuilder<IN, OUT, L, O> {
    /// Sets the [`LossFunction`] used by the [`NNTrainer`].
    pub fn loss_function<L2>(self, loss_function: L2) -> NNTrainerBuilder<IN, OUT, L2, O>
    where L2: LossFunction<OUT> {
        NNTrainerBuilder { loss_function, ..self }
    }

    /// Sets the [`Optimizer`] used by the [`NNTrainer`].
    pub fn optimizer<O2>(self, optimizer: O2) -> NNTrainerBuilder<IN, OUT, L, O2>
    where O2: OptimizerValues {
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
    pub fn new_clip_gradient_norm(self, max_norm: f64, norm_type: Norm) -> Self {
        let clip_grad_norm = ClipGradientNorm::new(norm_type, max_norm);
        self.clip_gradient_norm(clip_grad_norm)
    }
}

impl<const IN: usize, const OUT: usize, EO, L, O> NNTrainerBuilder<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: OptimizerValues,
{
    /// Consumes the Builder to create a new [`NNTrainer`].
    pub fn build(self) -> NNTrainer<IN, OUT, L, O::Optimizer> {
        let NNTrainerBuilder {
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
        } = self;
        let optimizer = optimizer.init_with_layers(network.get_layers()).into();

        if None == clip_gradient_norm {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        NNTrainer::new(network, loss_function, optimizer, retain_gradient, clip_gradient_norm)
    }
}
