use crate::prelude::*;

pub struct TrainableNeuralNetworkBuilder<const IN: usize, const OUT: usize, L>
where L: LossFunction<OUT>
{
    network: NeuralNetwork<IN, OUT>,
    loss_function: L,
    optimizer: Optimizer,
    retain_gradient: bool,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> TrainableNeuralNetworkBuilder<IN, OUT, HalfSquaredError> {
    /// `loss_function`: [`HalfSquaredError`]
    /// `retain_gradient`: `false`
    /// `optimizer`: Default [`GradientDescent`]
    /// `clip_gradient_norm`: [`None`]
    pub fn defaults(network: NeuralNetwork<IN, OUT>) -> Self {
        TrainableNeuralNetworkBuilder {
            network,
            loss_function: HalfSquaredError,
            retain_gradient: false,
            optimizer: Optimizer::default_gradient_descent(),
            clip_gradient_norm: None,
        }
    }
}

impl<const IN: usize, const OUT: usize, EO, L> TrainableNeuralNetworkBuilder<IN, OUT, L>
where L: LossFunction<OUT, ExpectedOutput = EO>
{
    pub fn error_function<NL: LossFunction<OUT>>(
        self,
        loss_function: NL,
    ) -> TrainableNeuralNetworkBuilder<IN, OUT, NL> {
        TrainableNeuralNetworkBuilder { loss_function, ..self }
    }

    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn sgd(self, sgd: GradientDescent) -> Self {
        self.optimizer(Optimizer::GradientDescent(sgd))
    }

    pub fn sgd_default(self) -> Self {
        self.sgd(GradientDescent::default())
    }

    pub fn new_sgd(self, learning_rate: f64) -> Self {
        self.sgd(GradientDescent { learning_rate })
    }

    pub fn adam(self, adam: Adam) -> Self {
        self.optimizer(Optimizer::Adam(adam))
    }

    pub fn adam_default(self, layers: &Vec<Layer>) -> Self {
        self.adam(Adam::default(layers))
    }

    pub fn new_adam(self, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let adam = Adam::new(learning_rate, beta1, beta2, epsilon, self.network.get_layers());
        self.adam(adam)
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

    pub fn build(self) -> TrainableNeuralNetwork<IN, OUT, L> {
        let TrainableNeuralNetworkBuilder {
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
        } = self;
        TrainableNeuralNetwork::new(
            network,
            loss_function,
            optimizer,
            retain_gradient,
            clip_gradient_norm,
        )
    }
}
