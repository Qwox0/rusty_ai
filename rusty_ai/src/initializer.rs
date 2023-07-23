use crate::prelude::*;
use rand::Rng;

/// see [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)
/// or [pytorch docs](https://pytorch.org/docs/stable/nn.init.html)
#[derive(Debug, Clone)]
pub enum Initializer<T> {
    Initialized(T),

    /// Initializes all values to the fixed value `self.0`
    Constant(f64),

    /// Uniform from start (`self.0`; inclusive) to end (`self.1`; exclusive)
    Uniform(f64, f64),

    /// `𝓝 (self.mean, self.std_dev^2)`
    Normal {
        mean: f64,
        std_dev: f64,
    },

    /// `𝓝 (0, 1)`
    ///
    /// Same as `Self::Normal { mean: 0.0, std_dev: 1.0 }` but faster.
    StandardNormal,

    /// also known as `Xavier Normal`.
    GlorotNormal,

    /// also known as `Xavier Uniform`.
    GlorotUniform,

    /// [TensorFlow Dense Layer docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    TensorFlowDefault,

    /// [Pytorch Linear Layer docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
    PytorchDefault,
}

pub type WeightsInitializer = Initializer<Matrix<f64>>;
pub type BiasInitializer = Initializer<LayerBias>;

impl<T> Initializer<T> {
    #[allow(non_snake_case)]
    pub const fn Ones() -> Self {
        Initializer::Constant(1.0)
    }

    #[allow(non_snake_case)]
    pub const fn Zeros() -> Self {
        Initializer::Constant(0.0)
    }
}

impl WeightsInitializer {
    pub fn init_weights(self, rng: &mut RngWrapper, inputs: usize, outputs: usize) -> Matrix<f64> {
        macro_rules! mat {
            ($iter:expr) => {
                Matrix::from_iter(inputs, outputs, $iter)
            };
        }
        use Initializer::*;
        match self {
            Initialized(x) => x,
            Constant(x) => Matrix::with_default(inputs, outputs, x),
            Uniform(low, high) => mat!(uniform(rng, low, high)),
            Normal { mean, std_dev } => mat!(normal(rng, mean, std_dev)),
            StandardNormal => mat!(std_normal(rng)),
            GlorotNormal => mat!(glorot_normal(rng, inputs, outputs)),
            GlorotUniform | TensorFlowDefault => mat!(glorot_uniform(rng, inputs, outputs)),
            PytorchDefault => mat!(pytorch_default(rng, inputs)),
        }
    }
}

impl BiasInitializer {
    pub fn init_bias(self, rng: &mut RngWrapper, inputs: usize, outputs: usize) -> LayerBias {
        macro_rules! bias {
            ($iter:expr) => {
                LayerBias::from_iter(outputs, $iter)
            };
        }
        use Initializer::*;
        match self {
            Initialized(x) => x,
            Constant(x) => LayerBias::from(vec![x; outputs]),
            TensorFlowDefault => LayerBias::from(vec![0.0; outputs]),
            Uniform(low, high) => bias!(uniform(rng, low, high)),
            Normal { mean, std_dev } => bias!(normal(rng, mean, std_dev)),
            StandardNormal => bias!(std_normal(rng)),
            GlorotNormal => bias!(glorot_normal(rng, inputs, outputs)),
            GlorotUniform => bias!(glorot_uniform(rng, inputs, outputs)),
            PytorchDefault => bias!(pytorch_default(rng, inputs)),
        }
    }
}

pub fn uniform<'a>(rng: &'a mut RngWrapper, low: f64, high: f64) -> impl Iterator<Item = f64> + 'a {
    rng.sample_iter(rand_distr::Uniform::new(low, high))
}

/// # Panics
/// Panics if `std_dev` is not finite.
pub fn normal<'a>(
    rng: &'a mut RngWrapper,
    mean: f64,
    std_dev: f64,
) -> impl Iterator<Item = f64> + 'a {
    rng.sample_iter(rand_distr::Normal::new(mean, std_dev).expect("standard deviation is finite"))
}

pub fn std_normal<'a>(rng: &'a mut RngWrapper) -> impl Iterator<Item = f64> + 'a {
    rng.sample_iter(rand_distr::StandardNormal)
}

pub fn glorot_uniform<'a>(
    rng: &'a mut RngWrapper,
    inputs: usize,
    outputs: usize,
) -> impl Iterator<Item = f64> + 'a {
    let x = (6.0 / (inputs + outputs) as f64).sqrt(); // TODO: gain
    uniform(rng, -x, x)
}

pub fn glorot_normal<'a>(
    rng: &'a mut RngWrapper,
    inputs: usize,
    outputs: usize,
) -> impl Iterator<Item = f64> + 'a {
    let std_dev = (2.0 / (inputs + outputs) as f64).sqrt(); // TODO: gain
    normal(rng, 0.0, std_dev)
}

pub fn pytorch_default<'a>(
    rng: &'a mut RngWrapper,
    inputs: usize,
) -> impl Iterator<Item = f64> + 'a {
    let x = (inputs as f64).recip().sqrt();
    uniform(rng, -x, x)
}
