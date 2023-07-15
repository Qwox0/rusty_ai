use crate::{
    prelude::{ActivationFn, Layer, LayerBias, Matrix},
    util::RngWrapper,
};
use rand::{distributions::Uniform, prelude::Distribution, Rng};

pub trait Initializer {
    fn init_weights(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> Matrix<f64>;
    fn init_bias(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> LayerBias;

    fn init_layer(
        &self,
        rng: &mut RngWrapper,
        inputs: usize,
        neurons: usize,
        activation_function: ActivationFn,
    ) -> Layer {
        let weights = self.init_weights(rng, inputs, neurons);
        let bias = self.init_bias(rng, inputs, neurons);
        Layer::new(weights, bias, activation_function)
    }
}

macro_rules! distr_initializer {
    ($type:ty : $get_distr:ident) => {
        impl Initializer for $type {
            fn init_weights(
                &self,
                rng: &mut RngWrapper,
                inputs: usize,
                neurons: usize,
            ) -> Matrix<f64> {
                let distr = self.$get_distr(inputs, neurons);
                Matrix::from_iter(inputs, neurons, rng.sample_iter(distr))
            }

            fn init_bias(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> LayerBias {
                let distr = self.$get_distr(inputs, neurons);
                LayerBias::from_iter(neurons, rng.sample_iter(distr))
            }
        }
    };
}

pub struct Rand<D: Distribution<f64>>(pub D);

impl<D: Distribution<f64>> Initializer for Rand<D> {
    fn init_weights(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> Matrix<f64> {
        Matrix::from_iter(inputs, neurons, rng.sample_iter(&self.0))
    }

    fn init_bias(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> LayerBias {
        LayerBias::from_iter(neurons, rng.sample_iter(&self.0))
    }
}

/// sets all parameters to the same value
pub struct SingleNum(f64);

impl Initializer for SingleNum {
    fn init_weights(&self, _rng: &mut RngWrapper, inputs: usize, neurons: usize) -> Matrix<f64> {
        Matrix::with_default(inputs, neurons, self.0)
    }

    fn init_bias(&self, _rng: &mut RngWrapper, _inputs: usize, neurons: usize) -> LayerBias {
        LayerBias::from(vec![self.0; neurons])
    }
}

type Zeros = SingleNum;
type Ones = SingleNum;
pub fn zeros() -> Zeros { SingleNum(0.0) }
pub fn ones() -> Ones { SingleNum(1.0) }

pub struct Custom<W: Initializer, B: Initializer> {
    weights_init: W,
    bias_init: B,
}

impl<W: Initializer, B: Initializer> Initializer for Custom<W, B> {
    fn init_weights(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> Matrix<f64> {
        self.weights_init.init_weights(rng, inputs, neurons)
    }

    fn init_bias(&self, rng: &mut RngWrapper, inputs: usize, neurons: usize) -> LayerBias {
        self.bias_init.init_bias(rng, inputs, neurons)
    }
}

/// also called Xavier uniform initializer.
/// [https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform]
pub struct GlorotUniform;
impl GlorotUniform {
    fn get_distr(&self, inputs: usize, neurons: usize) -> impl Distribution<f64> {
        let x = (6.0 / (inputs + neurons) as f64).sqrt();
        Uniform::from(-x..x)
    }
}
distr_initializer! { GlorotUniform: get_distr }

/// [https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense]
#[allow(non_snake_case)]
pub fn TensorFlowDefault() -> Custom<GlorotUniform, Zeros> {
    Custom { weights_init: GlorotUniform, bias_init: zeros() }
}

/// [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html]
pub struct PytorchDefault;
impl PytorchDefault {
    fn get_distr(&self, inputs: usize, _neurons: usize) -> impl Distribution<f64> {
        let x = (inputs as f64).recip().sqrt();
        Uniform::from(-x..x)
    }
}
distr_initializer! { PytorchDefault: get_distr }
