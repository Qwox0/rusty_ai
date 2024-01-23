use super::NNTrainer;
use crate::{
    clip_gradient_norm::ClipGradientNorm, loss_function::LossFunction, nn::NNComponent, norm::Norm,
    optimizer::Optimizer, NN,
};
#[allow(unused_imports)]
use const_tensor::{Element, Float, Shape};
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
pub struct NNTrainerBuilder<X, L, O, NN> {
    network: NN,
    loss_function: L,
    optimizer: O,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm<X>>,
}

impl<X, IN, OUT, C> NNTrainerBuilder<X, NoLossFunction, NoOptimizer, NN<X, IN, OUT, C>>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    C: NNComponent<X, IN, OUT>,
{
    /// # Defaults
    ///
    /// `retain_gradient`: `false`
    /// `clip_gradient_norm`: [`None`]
    /// `training_threads`: [`std::thread::available_parallelism`]
    pub fn new(network: NN<X, IN, OUT, C>) -> Self {
        NNTrainerBuilder {
            network,
            loss_function: NoLossFunction,
            retain_gradient: false,
            optimizer: NoOptimizer,
            clip_gradient_norm: None,
        }
    }
}

impl<X: Element, IN, OUT: Shape, L, O, C> NNTrainerBuilder<X, L, O, NN<X, IN, OUT, C>> {
    /// Sets the [`LossFunction`] used by the [`NNTrainer`].
    pub fn loss_function<L2>(
        self,
        loss_function: L2,
    ) -> NNTrainerBuilder<X, L2, O, NN<X, IN, OUT, C>>
    where
        L2: LossFunction<X, OUT>,
    {
        NNTrainerBuilder { loss_function, ..self }
    }
}

impl<X: Element, L, O, NN> NNTrainerBuilder<X, L, O, NN> {
    /// Sets the [`Optimizer`] used by the [`NNTrainer`].
    pub fn optimizer<O2>(self, optimizer: O2) -> NNTrainerBuilder<X, L, O2, NN>
    where O2: Optimizer<X> {
        NNTrainerBuilder { optimizer, ..self }
    }

    /// Sets whether the [`NNTrainer`] keeps or resets the [`Gradient`] between training steps.
    pub fn retain_gradient(mut self, retain_gradient: bool) -> Self {
        self.retain_gradient = retain_gradient;
        self
    }

    /// Activates and sets a [`ClipGradientNorm`].
    pub fn clip_gradient_norm(mut self, clip_gradient_norm: ClipGradientNorm<X>) -> Self {
        let _ = self.clip_gradient_norm.insert(clip_gradient_norm);
        self
    }
}

impl<X: Float, L, O, NN> NNTrainerBuilder<X, L, O, NN> {
    /// Activates and sets a new [`ClipGradientNorm`] which is created from the parameters.
    pub fn new_clip_gradient_norm(self, max_norm: X, norm_type: Norm) -> Self {
        let clip_grad_norm = ClipGradientNorm::new(norm_type, max_norm);
        self.clip_gradient_norm(clip_grad_norm)
    }
}

impl<X, IN, OUT, L, EO, O, C> NNTrainerBuilder<X, L, O, NN<X, IN, OUT, C>>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: Optimizer<X>,
    C: NNComponent<X, IN, OUT>,
{
    /// Consumes the Builder to create a new [`NNTrainer`].
    pub fn build(self) -> NNTrainer<X, IN, OUT, L, O, C> {
        let NNTrainerBuilder {
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
        } = self;
        if clip_gradient_norm == None {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        NNTrainer::new(network, loss_function, optimizer, retain_gradient, clip_gradient_norm)
    }
}
