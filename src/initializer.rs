//! # Initializer module

use const_tensor::{Element, Float, MultidimensionalOwned, Num, Shape, Tensor};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use std::ops::Range;

/// trait for initializing tensors.
pub trait Initializer<X: Element, S: Shape> {
    /// initialize a tensor from the inputs and outputs of a nn layer.
    fn init(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> Tensor<X, S>;
}

// for already initialized tensors
impl<X: Element, S: Shape> Initializer<X, S> for Tensor<X, S> {
    fn init(self, _rng: &mut impl Rng, _inputs: usize, _outputs: usize) -> Tensor<X, S> {
        self
    }
}

/// Initializes all values with the fixed value `self.0`
pub struct Constant<X>(pub X);

impl<X: Element, S: Shape> Initializer<X, S> for Constant<X> {
    fn init(self, _rng: &mut impl Rng, _inputs: usize, _outputs: usize) -> Tensor<X, S> {
        Tensor::full(self.0)
    }
}

/// Initializes all values with the value zero.
#[allow(non_snake_case)]
pub fn Zeros<X: Num>() -> Constant<X> {
    Constant(X::ZERO)
}

/// Initializes all values with the value one.
#[allow(non_snake_case)]
pub fn Ones<X: Num>() -> Constant<X> {
    Constant(X::ONE)
}

/// Uniform
pub struct Uniform<X>(pub Range<X>);

impl<X: Element, S: Shape> Initializer<X, S> for Uniform<X>
where X: SampleUniform
{
    fn init(self, rng: &mut impl Rng, _inputs: usize, _outputs: usize) -> Tensor<X, S> {
        Tensor::from_iter(rng.sample_iter(rand_distr::Uniform::from(self.0)))
    }
}

/// `𝓝 (self.mean, self.std_dev^2)`
pub struct Normal<X> {
    /// mean of the normal distribution
    pub mean: X,
    /// standard deviation of the normal distribution
    pub std_dev: X,
}

impl<X: Float, S: Shape> Initializer<X, S> for Normal<X>
where rand_distr::StandardNormal: rand_distr::Distribution<X>
{
    /// # Panics
    /// Panics if `std_dev` is not finite.
    fn init(self, rng: &mut impl Rng, _inputs: usize, _outputs: usize) -> Tensor<X, S> {
        Tensor::from_iter(rng.sample_iter(
            rand_distr::Normal::new(self.mean, self.std_dev).expect("standard deviation is finite"),
        ))
    }
}

/// `𝓝 (0, 1)`
///
/// Same as `Self::Normal { mean: 0.0, std_dev: 1.0 }` but faster.
pub struct StandardNormal;

impl<X: Element, S: Shape> Initializer<X, S> for StandardNormal
where rand_distr::StandardNormal: rand_distr::Distribution<X>
{
    fn init(self, rng: &mut impl Rng, _inputs: usize, _outputs: usize) -> Tensor<X, S> {
        Tensor::from_iter(rng.sample_iter(rand_distr::StandardNormal))
    }
}

/// also known as `Xavier Normal`.
pub struct GlorotNormal;

impl<X: Float, S: Shape> Initializer<X, S> for GlorotNormal
where rand_distr::StandardNormal: rand_distr::Distribution<X>
{
    fn init(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> Tensor<X, S> {
        let std_dev = (X::lit(2) / (inputs + outputs).cast()).sqrt(); // TODO: gain
        Normal { mean: X::ZERO, std_dev }.init(rng, inputs, outputs)
    }
}

/// also known as `Xavier Uniform`.
pub struct GlorotUniform;

impl<X: Float, S: Shape> Initializer<X, S> for GlorotUniform
where X: SampleUniform
{
    fn init(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> Tensor<X, S> {
        let x = (X::lit(6) / (inputs + outputs).cast()).sqrt(); // TODO: gain
        Uniform(-x..x).init(rng, inputs, outputs)
    }
}

/// [TensorFlow Dense Layer docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
//pub struct TensorFlowDefault;
//impl<X: Element, S: Shape> Initializer<X, S> for TensorFlowDefault {}

/// [Pytorch Linear Layer docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
pub struct PytorchDefault;

impl<X: Float, S: Shape> Initializer<X, S> for PytorchDefault
where X: SampleUniform
{
    fn init(self, rng: &mut impl Rng, inputs: usize, outputs: usize) -> Tensor<X, S> {
        let x = inputs.cast::<X>().recip().sqrt();
        Uniform(-x..x).init(rng, inputs, outputs)
    }
}
