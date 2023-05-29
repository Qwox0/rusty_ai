use super::{Buildable, RandomMarker};
use crate::{
    prelude::{ActivationFn, Matrix},
    util::RngWrapper,
};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

pub struct Incomplete<D: Distribution<f64>> {
    pub neurons: usize,
    pub distr: D,
}

pub struct RandomWeights<D: Distribution<f64>> {
    pub inputs: usize,
    pub neurons: usize,
    pub distr: D,
}

#[derive(Debug)]
pub struct WeightsInitialized(pub Matrix<f64>);

pub type DefaultIncomplete = Incomplete<Uniform<f64>>;
impl DefaultIncomplete {
    pub fn default(neurons: usize) -> DefaultIncomplete {
        Incomplete {
            neurons,
            distr: Uniform::from(0.0..1.0),
        }
    }
}

pub trait WeightsMarker {
    fn get_neuron_count(&self) -> usize;
}

impl<D: Distribution<f64>> WeightsMarker for Incomplete<D> {
    fn get_neuron_count(&self) -> usize {
        self.neurons
    }
}

impl<D: Distribution<f64>> WeightsMarker for RandomWeights<D> {
    fn get_neuron_count(&self) -> usize {
        self.neurons
    }
}

impl WeightsMarker for WeightsInitialized {
    fn get_neuron_count(&self) -> usize {
        self.0.get_height()
    }
}

impl<D: Distribution<f64>> RandomWeights<D> {
    pub fn from_incomplete(weights: Incomplete<D>, inputs: usize) -> Self {
        let Incomplete { neurons, distr } = weights;
        RandomWeights {
            inputs,
            neurons,
            distr,
        }
    }
}

impl<D: Distribution<f64>> Buildable for RandomWeights<D> {
    type OUT = Matrix<f64>;
    fn build(self, rng: &mut crate::util::RngWrapper) -> Self::OUT {
        Matrix::random(self.inputs, self.neurons, rng, &self.distr)
    }

    fn clone_build(&mut self, rng: &mut RngWrapper) -> Self::OUT {
        Matrix::random(self.inputs, self.neurons, rng, &self.distr)
    }
}

impl Buildable for WeightsInitialized {
    type OUT = Matrix<f64>;
    #[inline]
    fn build(self, _rng: &mut RngWrapper) -> Matrix<f64> {
        self.0
    }

    fn clone_build(&mut self, _rng: &mut RngWrapper) -> Self::OUT {
        self.0.clone()
    }
}
