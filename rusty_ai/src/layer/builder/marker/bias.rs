use super::{Buildable, RandomMarker};
use crate::{prelude::LayerBias, util::RngWrapper};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
pub enum BiasType {
    OnePerLayer,
    OnePerNeuron(usize),
}

// Markers
#[derive(Debug)]
pub struct RandomBias<D: Distribution<f64>> {
    pub bias_type: BiasType,
    pub distr: D,
}

pub struct BiasInitialized(pub LayerBias);

pub type DefaultBias = RandomBias<Uniform<f64>>;
impl DefaultBias {
    pub fn default(neurons: usize) -> DefaultBias {
        RandomBias {
            bias_type: BiasType::OnePerNeuron(neurons),
            distr: Uniform::from(0.0..1.0),
        }
    }
}

pub trait BiasMarker {
    fn get_bias_type(&self) -> BiasType;
}

impl<D: Distribution<f64>> BiasMarker for RandomBias<D> {
    fn get_bias_type(&self) -> BiasType {
        self.bias_type.clone()
    }
}

impl BiasMarker for BiasInitialized {
    fn get_bias_type(&self) -> BiasType {
        match &self.0 {
            LayerBias::OnePerLayer(_) => BiasType::OnePerLayer,
            LayerBias::OnePerNeuron(vec) => BiasType::OnePerNeuron(vec.len()),
        }
    }
}

impl<D: Distribution<f64>> Buildable for RandomBias<D> {
    type OUT = LayerBias;
    fn build(mut self, rng: &mut crate::util::RngWrapper) -> Self::OUT {
        self.clone_build(rng)
    }

    fn clone_build(&mut self, rng: &mut RngWrapper) -> Self::OUT {
        match self.bias_type {
            BiasType::OnePerLayer => LayerBias::OnePerLayer(rng.sample(&self.distr)),
            BiasType::OnePerNeuron(neurons) => {
                LayerBias::OnePerNeuron(rng.sample_iter(&self.distr).take(neurons).collect())
            }
        }
    }
}

impl Buildable for BiasInitialized {
    type OUT = LayerBias;
    fn build(self, _rng: &mut RngWrapper) -> LayerBias {
        self.0
    }

    fn clone_build(&mut self, rng: &mut RngWrapper) -> Self::OUT {
        self.0.clone()
    }
}
