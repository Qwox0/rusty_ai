use super::NNTrainer;
use crate::{
    clip_gradient_norm::ClipGradientNorm, loss_function::LossFunction, norm::Norm,
    optimizer::Optimizer, NN,
};
#[allow(unused_imports)]
use const_tensor::{Element, Float, Shape};
use markers::*;
use std::marker::PhantomData;

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
pub struct NNTrainerBuilder<X, IN, OUT, L, O, NN_> {
    network: NN_,
    _dim: PhantomData<(IN, OUT)>,
    loss_function: L,
    optimizer: O,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm<X>>,
}

impl<X, IN, OUT, NN_> NNTrainerBuilder<X, IN, OUT, NoLossFunction, NoOptimizer, NN_>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    NN_: NN<X, IN, OUT>,
{
    /// # Defaults
    ///
    /// `retain_gradient`: `false`
    /// `clip_gradient_norm`: [`None`]
    /// `training_threads`: [`std::thread::available_parallelism`]
    pub fn new(network: NN_) -> Self {
        NNTrainerBuilder {
            network,
            _dim: PhantomData,
            loss_function: NoLossFunction,
            retain_gradient: false,
            optimizer: NoOptimizer,
            clip_gradient_norm: None,
        }
    }
}

impl<X: Element, IN: Shape, OUT: Shape, L, O, NN_: NN<X, IN, OUT>>
    NNTrainerBuilder<X, IN, OUT, L, O, NN_>
{
    /// Sets the [`LossFunction`] used by the [`NNTrainer`].
    pub fn loss_function<L2>(self, loss_function: L2) -> NNTrainerBuilder<X, IN, OUT, L2, O, NN_>
    where L2: LossFunction<X, OUT> {
        NNTrainerBuilder { loss_function, ..self }
    }

    /// Sets the [`Optimizer`] used by the [`NNTrainer`].
    pub fn optimizer<O2>(self, optimizer: O2) -> NNTrainerBuilder<X, IN, OUT, L, O2, NN_>
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

impl<X: Float, IN: Shape, OUT: Shape, L, O, NN_: NN<X, IN, OUT>>
    NNTrainerBuilder<X, IN, OUT, L, O, NN_>
{
    /// Activates and sets a new [`ClipGradientNorm`] which is created from the parameters.
    pub fn new_clip_gradient_norm(self, max_norm: X, norm_type: Norm) -> Self {
        let clip_grad_norm = ClipGradientNorm::new(norm_type, max_norm);
        self.clip_gradient_norm(clip_grad_norm)
    }
}

impl<X, IN, OUT, L, EO, O, NN_> NNTrainerBuilder<X, IN, OUT, L, O, NN_>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: Optimizer<X>,
    NN_: NN<X, IN, OUT>,
{
    /// Consumes the Builder to create a new [`NNTrainer`].
    pub fn build(self) -> NNTrainer<X, IN, OUT, L, O, NN_> {
        let NNTrainerBuilder {
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
            ..
        } = self;
        if clip_gradient_norm == None {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        NNTrainer::new(network, loss_function, optimizer, retain_gradient, clip_gradient_norm)
    }
}
