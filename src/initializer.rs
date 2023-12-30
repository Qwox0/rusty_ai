use crate::{bias::LayerBias, matrix::Matrix};
use matrix::{Float, Num};
use rand::Rng;

/// see [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)
/// or [pytorch docs](https://pytorch.org/docs/stable/nn.init.html)
#[derive(Debug, Clone)]
pub enum Initializer<X, T> {
    /// Fixed value
    Initialized(T),

    /// Initializes all values with the fixed value `self.0`
    Constant(X),

    /// Uniform from start (`self.0`; inclusive) to end (`self.1`; exclusive)
    Uniform(X, X),

    /// `𝓝 (self.mean, self.std_dev^2)`
    Normal {
        /// mean of the normal distribution
        mean: X,
        /// standard deviation of the normal distribution
        std_dev: X,
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

// pub type DataInitializer<const DIM: usize> = Initializer<[f64; DIM]>;

impl<X: Num, T> Initializer<X, T> {
    /// Initializes all values with the fixed value `1`.
    #[allow(non_upper_case_globals)]
    pub const Ones: Self = Initializer::Constant(X::ONE);
    /// Initializes all values with the fixed value `0`.
    #[allow(non_upper_case_globals)]
    pub const Zeros: Self = Initializer::Constant(X::ZERO);
}

impl<X: Float> Initializer<X, Matrix<X>>
where rand_distr::StandardNormal: rand_distr::Distribution<X>
{
    /// Uses `self` to create a weights [`Matrix`].
    pub fn init_weights(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> Matrix<X> {
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

impl<X: Float> Initializer<X, LayerBias<X>>
where rand_distr::StandardNormal: rand_distr::Distribution<X>
{
    /// Uses `self` to create a weights [`LayerBias<X>`].
    pub fn init_bias(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> LayerBias<X> {
        macro_rules! bias {
            ($iter:expr) => {
                LayerBias::from_iter(outputs, $iter)
            };
        }
        use Initializer::*;
        match self {
            Initialized(x) => x,
            Constant(x) => LayerBias::from(vec![x; outputs]),
            TensorFlowDefault => LayerBias::from(vec![X::zero(); outputs]),
            Uniform(low, high) => bias!(uniform(rng, low, high)),
            Normal { mean, std_dev } => bias!(normal(rng, mean, std_dev)),
            StandardNormal => bias!(std_normal(rng)),
            GlorotNormal => bias!(glorot_normal(rng, inputs, outputs)),
            GlorotUniform => bias!(glorot_uniform(rng, inputs, outputs)),
            PytorchDefault => bias!(pytorch_default(rng, inputs)),
        }
    }
}

fn uniform<'a, X: Float>(rng: &'a mut impl Rng, low: X, high: X) -> impl Iterator<Item = X> + 'a {
    rng.sample_iter(rand_distr::Uniform::new(low, high))
}

/// # Panics
/// Panics if `std_dev` is not finite.
fn normal<'a, X: Float>(
    rng: &'a mut impl Rng,
    mean: X,
    std_dev: X,
) -> impl Iterator<Item = X> + 'a
where
    rand_distr::StandardNormal: rand_distr::Distribution<X>,
{
    rng.sample_iter(rand_distr::Normal::new(mean, std_dev).expect("standard deviation is finite"))
}

fn std_normal<'a, X: 'a>(rng: &'a mut impl Rng) -> impl Iterator<Item = X> + 'a
where rand_distr::StandardNormal: rand_distr::Distribution<X> {
    rng.sample_iter(rand_distr::StandardNormal)
}

fn glorot_uniform<'a, X: Float>(
    rng: &'a mut impl Rng,
    inputs: usize,
    outputs: usize,
) -> impl Iterator<Item = X> + 'a {
    let x = (6.0.cast::<X>() / (inputs + outputs).cast()).sqrt(); // TODO: gain
    uniform(rng, -x, x)
}

fn glorot_normal<'a, X: Float>(
    rng: &'a mut impl Rng,
    inputs: usize,
    outputs: usize,
) -> impl Iterator<Item = X> + 'a
where
    rand_distr::StandardNormal: rand_distr::Distribution<X>,
{
    let std_dev = (2.0.cast::<X>() / (inputs + outputs).cast()).sqrt(); // TODO: gain
    normal(rng, X::zero(), std_dev)
}

fn pytorch_default<'a, X: Float>(
    rng: &'a mut impl Rng,
    inputs: usize,
) -> impl Iterator<Item = X> + 'a {
    let x = inputs.cast::<X>().recip().sqrt();
    uniform(rng, -x, x)
}
