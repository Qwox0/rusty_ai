mod marker;
pub use marker::*;

use super::{Layer, LayerBias};
use crate::{
    prelude::{ActivationFn, Matrix},
    util::RngWrapper,
};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

const DEFAULT_ACTIVATION_FN: ActivationFn = ActivationFn::default_relu();

#[derive(Debug, Clone)]
pub struct LayerBuilder<W: WeightsMarker, B: BiasMarker> {
    weights: W,
    bias: B,
    activation_function: Option<ActivationFn>,
    rng: RngWrapper,
}

impl LayerBuilder<DefaultIncomplete, DefaultBias> {
    pub fn neurons(neurons: usize) -> LayerBuilder<DefaultIncomplete, DefaultBias> {
        LayerBuilder {
            weights: Incomplete::default(neurons),
            bias: RandomBias::default(neurons),
            activation_function: None,
            rng: RngWrapper::default(),
        }
    }
}

impl<D: Distribution<f64>, B: BiasMarker> LayerBuilder<Incomplete<D>, B> {
    pub fn inputs(self, inputs: usize) -> LayerBuilder<RandomWeights<D>, B> {
        let weights = RandomWeights::from_incomplete(self.weights, inputs);
        LayerBuilder { weights, ..self }
    }

    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        assert_eq!(
            self.weights.get_neuron_count(),
            weights.get_height(),
            "Weights doesn't match previously set neuron count/height."
        );
        self.weights_unchecked(weights)
    }

    pub fn random_weights<ND: Distribution<f64>>(
        self,
        distr: ND,
    ) -> LayerBuilder<Incomplete<ND>, B> {
        let weights = Incomplete {
            distr,
            ..self.weights
        };
        LayerBuilder { weights, ..self }
    }
}

impl LayerBuilder<RandomWeights<Uniform<f64>>, DefaultBias> {
    pub fn with_inputs(
        inputs: usize,
        neurons: usize,
    ) -> LayerBuilder<RandomWeights<Uniform<f64>>, DefaultBias> {
        LayerBuilder::neurons(neurons).inputs(inputs)
    }
}

impl<D: Distribution<f64>, B: BiasMarker> LayerBuilder<RandomWeights<D>, B> {
    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        assert_eq!(
            (self.weights.inputs, self.weights.neurons),
            weights.get_dimensions(),
            "Weights doesn't match previously set dimensions."
        );
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<WeightsInitialized, DefaultBias> {
    pub fn with_weights(weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, DefaultBias> {
        LayerBuilder::neurons(weights.get_height()).weights_unchecked(weights)
    }
}

impl<W: WeightsMarker, B: BiasMarker> LayerBuilder<W, B> {
    fn weights_unchecked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        let weights = WeightsInitialized(weights);
        LayerBuilder { weights, ..self }
    }

    pub fn bias(self, bias: LayerBias) -> LayerBuilder<W, BiasInitialized> {
        if let Some(neuron_count) = bias.get_neuron_count() {
            assert_eq!(
                neuron_count,
                self.weights.get_neuron_count(),
                "Bias doesn't match previously set neuron count."
            )
        }

        let bias = BiasInitialized(bias);
        LayerBuilder { bias, ..self }
    }

    pub fn random_bias<D: Distribution<f64>>(self, distr: D) -> LayerBuilder<W, RandomBias<D>> {
        let bias = RandomBias {
            bias_type: self.bias.get_bias_type(),
            distr,
        };
        LayerBuilder { bias, ..self }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = RngWrapper::Seeded(StdRng::seed_from_u64(seed));
        self
    }

    pub fn activation_function(mut self, act_fn: ActivationFn) -> Self {
        self.activation_function = Some(act_fn);
        self
    }
}

impl<W, B> LayerBuilder<W, B>
where
    W: WeightsMarker + Buildable<OUT = Matrix<f64>>,
    B: BiasMarker + Buildable<OUT = LayerBias>,
{
    pub fn build_clone(&mut self) -> Layer {
        Layer::new(
            self.weights.clone_build(&mut self.rng),
            self.bias.clone_build(&mut self.rng),
            self.activation_function.unwrap_or(DEFAULT_ACTIVATION_FN),
        )
    }

    pub fn build(mut self) -> Layer {
        Layer::new(
            self.weights.build(&mut self.rng),
            self.bias.build(&mut self.rng),
            self.activation_function.unwrap_or(DEFAULT_ACTIVATION_FN),
        )
    }
}

pub trait LayerOrLayerBuilder {
    /// Consumes self to create a [`Layer`].
    /// # Panics
    /// Panics if `inputs` doesn't match the previously set input value (if it exists)
    fn as_layer_with_inputs(self, inputs: usize) -> Layer;
}

impl<D, B> LayerOrLayerBuilder for LayerBuilder<Incomplete<D>, B>
where
    D: Distribution<f64>,
    B: BiasMarker + Buildable<OUT = LayerBias>,
{
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        self.inputs(inputs).build()
    }
}

impl<W, B> LayerOrLayerBuilder for LayerBuilder<W, B>
where
    W: WeightsMarker + Buildable<OUT = Matrix<f64>>,
    B: BiasMarker + Buildable<OUT = LayerBias>,
{
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        let layer = self.build();
        assert_eq!(
            inputs,
            layer.get_input_count(),
            "input count doesn't match previously set value"
        );
        layer
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn old_with_weights_panics() {
        let weights = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        LayerBuilder::neurons(0).weights_unchecked(weights).build();
    }

    #[test]
    fn with_weights() {
        let weights = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        LayerBuilder::with_weights(weights).build();
    }
}
