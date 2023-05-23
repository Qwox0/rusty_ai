use super::{Layer, LayerBias};
use crate::prelude::{ActivationFn, Matrix};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

pub trait LayerOrLayerBuilder {
    /// Consumes self to create a [`Layer`].
    /// # Panics
    /// Panics if `inputs` doesn't match the previously set input value (if it exists)
    fn as_layer_with_inputs(self, inputs: usize) -> Layer;
}

// Weights Markers
pub struct Incomplete {
    neurons: usize,
}
pub struct IncompleteRandom<D: Distribution<f64>> {
    neurons: usize,
    distr: D,
    seed: Option<u64>,
}
pub struct RandomWeights<D: Distribution<f64>> {
    inputs: usize,
    neurons: usize,
    distr: D,
    seed: Option<u64>,
}
#[derive(Debug)]
pub struct WeightsInitialized(Matrix<f64>);

type DefaultRandomWeights = RandomWeights<Uniform<f64>>;
impl DefaultRandomWeights {
    fn default(inputs: usize, neurons: usize) -> DefaultRandomWeights {
        RandomWeights {
            inputs,
            neurons,
            distr: Uniform::from(0.0..1.0),
            seed: None,
        }
    }
}

pub trait BuildWeights {
    fn build_weights(self) -> Matrix<f64>;
}
impl<D: Distribution<f64>> RandomWeights<D> {
    fn sample(&self, rng: &mut impl Rng) -> Matrix<f64> {
        Matrix::random(self.inputs, self.neurons, rng, &self.distr)
    }
}
impl<D: Distribution<f64>> BuildWeights for RandomWeights<D> {
    fn build_weights(self) -> Matrix<f64> {
        match self.seed {
            Some(seed) => self.sample(&mut StdRng::seed_from_u64(seed)),
            None => self.sample(&mut rand::thread_rng()),
        }
    }
}
impl BuildWeights for WeightsInitialized {
    #[inline]
    fn build_weights(self) -> Matrix<f64> {
        self.0
    }
}

// Bias Markers
#[derive(Debug)]
pub struct RandomBias<D: Distribution<f64>> {
    bias_type: BiasType,
    distr: D,
    seed: Option<u64>,
}
#[derive(Debug, Clone)]
pub enum BiasType {
    OnePerLayer,
    OnePerNeuron(usize),
}
pub struct BiasInitialized(LayerBias);

type DefaultRandomBias = RandomBias<Uniform<f64>>;
impl DefaultRandomBias {
    fn default(neurons: usize) -> DefaultRandomBias {
        RandomBias {
            bias_type: BiasType::OnePerNeuron(neurons),
            distr: Uniform::from(0.0..1.0),
            seed: None,
        }
    }
}

pub trait BiasMarker {
    fn build_bias(self) -> LayerBias;
    fn get_bias_type(&self) -> BiasType;
}
impl<D: Distribution<f64>> RandomBias<D> {
    fn sample(&self, mut rng: impl Rng) -> LayerBias {
        match self.bias_type {
            BiasType::OnePerLayer => LayerBias::OnePerLayer(rng.sample(&self.distr)),
            BiasType::OnePerNeuron(count) => {
                LayerBias::OnePerNeuron(rng.sample_iter(&self.distr).take(count).collect())
            }
        }
    }
}
impl<D: Distribution<f64>> BiasMarker for RandomBias<D> {
    fn build_bias(self) -> LayerBias {
        match self.seed {
            Some(seed) => self.sample(StdRng::seed_from_u64(seed)),
            None => self.sample(rand::thread_rng()),
        }
    }

    fn get_bias_type(&self) -> BiasType {
        self.bias_type.clone()
    }
}
impl BiasMarker for BiasInitialized {
    fn build_bias(self) -> LayerBias {
        self.0
    }

    fn get_bias_type(&self) -> BiasType {
        match &self.0 {
            LayerBias::OnePerLayer(_) => BiasType::OnePerLayer,
            LayerBias::OnePerNeuron(vec) => BiasType::OnePerNeuron(vec.len()),
        }
    }
}

const DEFAULT_ACTIVATION_FN: ActivationFn = ActivationFn::default_relu();

#[derive(Debug, Clone)]
pub struct LayerBuilder<W, B> {
    weights: W,
    bias: B,
    activation_function: Option<ActivationFn>,
}

impl LayerBuilder<Incomplete, DefaultRandomBias> {
    pub fn new(neurons: usize) -> LayerBuilder<Incomplete, DefaultRandomBias> {
        LayerBuilder {
            weights: Incomplete { neurons },
            bias: RandomBias::default(neurons),
            activation_function: None,
        }
    }
}

impl<B: BiasMarker> LayerBuilder<Incomplete, B> {
    pub fn inputs(self, inputs: usize) -> LayerBuilder<DefaultRandomWeights, B> {
        let Incomplete { neurons } = self.weights;
        let weights = RandomWeights::default(inputs, neurons);
        LayerBuilder { weights, ..self }
    }

    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        let Incomplete { neurons } = self.weights;
        assert_eq!(neurons, weights.get_height());
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<DefaultRandomWeights, DefaultRandomBias> {
    pub fn with_inputs(
        inputs: usize,
        neurons: usize,
    ) -> LayerBuilder<DefaultRandomWeights, DefaultRandomBias> {
        LayerBuilder::new(neurons).inputs(inputs)
    }
}

impl<D: Distribution<f64>, B: BiasMarker> LayerBuilder<RandomWeights<D>, B> {
    /// initializes `weights` to a specific value.
    /// # Panics
    /// This panics if the dimensions of `weights` don't match previously set layer dimensions.
    pub fn weights_checked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        assert_eq!(
            (self.weights.inputs, self.weights.neurons),
            weights.get_dimensions()
        );
        self.weights_unchecked(weights)
    }
}

impl LayerBuilder<WeightsInitialized, DefaultRandomBias> {
    pub fn with_weights(
        weights: Matrix<f64>,
    ) -> LayerBuilder<WeightsInitialized, DefaultRandomBias> {
        LayerBuilder::new(weights.get_height()).weights_unchecked(weights)
    }
}

impl<W, B: BiasMarker> LayerBuilder<W, B> {
    /// Initializes layer weights to the specific value `bias`. This ignores any previously set
    /// layer dimensions.
    /// This doesn't change the bias which might cause the weights and bias to have different
    /// neuron counts. This difference lead to a panic when calling `build`.
    pub fn weights_unchecked(self, weights: Matrix<f64>) -> LayerBuilder<WeightsInitialized, B> {
        let weights = WeightsInitialized(weights);
        LayerBuilder { weights, ..self }
    }

    /*
    pub fn random_weights<D: Distribution<f64>>(self, distr: D) -> LayerBuilder<WeightsRandom<D>, B> {
        let weights = WeightsRandom { inputs: (), neurons: (), distr: (), seed: None }
        LayerBuilder { weights, ..self }
    }*/

    /// Initializes layer bias to the specific value `bias`.
    pub fn bias(self, bias: LayerBias) -> LayerBuilder<W, BiasInitialized> {
        let bias = BiasInitialized(bias);
        LayerBuilder { bias, ..self }
    }

    pub fn random_bias<D: Distribution<f64>>(self, distr: D) -> LayerBuilder<W, RandomBias<D>> {
        let bias = RandomBias {
            bias_type: self.bias.get_bias_type(),
            distr,
            seed: None,
        };
        LayerBuilder { bias, ..self }
    }

    pub fn activation_function(mut self, act_fn: ActivationFn) -> Self {
        self.activation_function = Some(act_fn);
        self
    }
}

impl<W: BuildWeights, B: BiasMarker> LayerBuilder<W, B> {
    pub fn build(self) -> Layer {
        Layer::new(
            self.weights.build_weights(),
            self.bias.build_bias(),
            self.activation_function.unwrap_or(DEFAULT_ACTIVATION_FN),
        )
    }
}

impl<B: BiasMarker> LayerOrLayerBuilder for LayerBuilder<Incomplete, B> {
    fn as_layer_with_inputs(self, inputs: usize) -> Layer {
        self.inputs(inputs).build()
    }
}

impl<W: BuildWeights, B: BiasMarker> LayerOrLayerBuilder for LayerBuilder<W, B> {
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
    use super::LayerBuilder;
    use crate::prelude::Matrix;

    #[test]
    #[should_panic]
    fn old_with_weights_panics() {
        let weights = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        LayerBuilder::new(0).weights_unchecked(weights).build();
    }

    #[test]
    fn with_weights() {
        let weights = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        LayerBuilder::with_weights(weights).build();
    }
}
