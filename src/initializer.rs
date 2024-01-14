use const_tensor::{Element, Float, Len, Num, Shape, Tensor};
use rand::Rng;

/// see [tensorflow docs](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)
/// or [pytorch docs](https://pytorch.org/docs/stable/nn.init.html)
#[derive(Debug, Clone)]
pub enum Initializer<X: Element, S: Shape> {
    /// Fixed value
    Initialized(Tensor<X, S>),

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
    //TensorFlowDefault,

    /// [Pytorch Linear Layer docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
    PytorchDefault,
}

// pub type DataInitializer<const DIM: usize> = Initializer<[f64; DIM]>;

impl<X: Num, S: Shape> Initializer<X, S> {
    /// Initializes all values with the fixed value `1`.
    #[allow(non_upper_case_globals)]
    pub const Ones: Self = Initializer::Constant(X::ONE);
    /// Initializes all values with the fixed value `0`.
    #[allow(non_upper_case_globals)]
    pub const Zeros: Self = Initializer::Constant(X::ZERO);
}

impl<F: Float, S: Shape> Initializer<F, S>
where rand_distr::StandardNormal: rand_distr::Distribution<F>
{
    /// Uses `self` to create a weights [`Matrix`].
    pub fn init<const LEN: usize>(
        self,
        rng: &mut impl Rng,
        inputs: usize,
        outputs: usize,
    ) -> Tensor<F, S>
    where
        S: Len<LEN>,
    {
        use Initializer as I;
        match self {
            I::Initialized(t) => t,
            I::Constant(x) => Tensor::full(x),
            I::Uniform(low, high) => Tensor::from_iter(uniform(rng, low, high)),
            I::Normal { mean, std_dev } => Tensor::from_iter(normal(rng, mean, std_dev)),
            I::StandardNormal => Tensor::from_iter(std_normal(rng)),
            I::GlorotNormal => Tensor::from_iter(glorot_normal(rng, inputs, outputs)),
            I::GlorotUniform => Tensor::from_iter(glorot_uniform(rng, inputs, outputs)),
            I::PytorchDefault => Tensor::from_iter(pytorch_default(rng, inputs)),
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
