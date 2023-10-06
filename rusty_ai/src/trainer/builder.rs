use crate::prelude::*;

pub struct NoLossFunction;
pub struct NoOptimizer;

pub struct NNTrainerBuilder<const IN: usize, const OUT: usize, L, O> {
    network: NeuralNetwork<IN, OUT>,
    loss_function: L,
    optimizer: O,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> NNTrainerBuilder<IN, OUT, NoLossFunction, NoOptimizer> {
    /// `retain_gradient`: `false`
    /// `optimizer`: Default [`GradientDescent`]
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
    pub fn loss_function<L2>(self, loss_function: L2) -> NNTrainerBuilder<IN, OUT, L2, O>
    where L2: LossFunction<OUT> {
        NNTrainerBuilder { loss_function, ..self }
    }

    pub fn optimizer<O2>(self, optimizer: O2) -> NNTrainerBuilder<IN, OUT, L, O2>
    where O2: OptimizerValues {
        NNTrainerBuilder { optimizer, ..self }
    }

    pub fn retain_gradient(mut self, retain_gradient: bool) -> Self {
        self.retain_gradient = retain_gradient;
        self
    }

    pub fn clip_gradient_norm(mut self, clip_gradient_norm: ClipGradientNorm) -> Self {
        let _ = self.clip_gradient_norm.insert(clip_gradient_norm);
        self
    }

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
