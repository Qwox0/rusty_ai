use rand::{distributions::DistIter, prelude::Distribution, Rng};

use super::AddBias;
use crate::{
    gradient::aliases::{BiasGradient, WeightedSumGradient},
    util::{
        EntryAdd, EntryDiv, EntryMul, EntrySub, Lerp, Randomize, RngWrapper, ScalarAdd, ScalarDiv,
        ScalarMul,
    },
};

#[derive(Debug, Clone)]
pub enum LayerBias {
    OnePerLayer(f64),
    OnePerNeuron(Vec<f64>),
}

impl LayerBias {
    pub fn new_singular(bias: f64) -> LayerBias {
        LayerBias::OnePerLayer(bias)
    }

    pub fn new_multiple(bias: Vec<f64>) -> LayerBias {
        LayerBias::OnePerNeuron(bias)
    }

    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter_singular(mut iter: impl Iterator<Item = f64>) -> LayerBias {
        LayerBias::new_singular(iter.next().unwrap())
    }

    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter_multiple(count: usize, iter: impl Iterator<Item = f64>) -> LayerBias {
        let vec: Vec<_> = iter.take(count).collect();
        assert_eq!(vec.len(), count);
        LayerBias::new_multiple(vec)
    }

    pub fn fill(mut self, value: f64) -> LayerBias {
        match &mut self {
            LayerBias::OnePerLayer(x) => *x = value,
            LayerBias::OnePerNeuron(vec) => vec.fill(value),
        }
        self
    }

    pub fn clone_with_zeros(&self) -> LayerBias {
        use LayerBias::*;
        match self {
            OnePerLayer(_) => OnePerLayer(0.0),
            OnePerNeuron(vec) => OnePerNeuron(vec![0.0; vec.len()]),
        }
    }

    pub fn new_matching_gradient(
        &self,
        weighted_sum_gradient: &WeightedSumGradient,
    ) -> BiasGradient {
        use LayerBias::*;
        match self {
            OnePerLayer(_) => OnePerLayer(weighted_sum_gradient.iter().sum()),
            OnePerNeuron(_) => OnePerNeuron(weighted_sum_gradient.clone()),
        }
    }

    pub fn get_neuron_count(&self) -> Option<usize> {
        match self {
            LayerBias::OnePerLayer(_) => None,
            LayerBias::OnePerNeuron(vec) => Some(vec.len()),
        }
    }

    pub fn sqare_entries_mut(&mut self) -> &mut LayerBias {
        match self {
            LayerBias::OnePerLayer(x) => *x *= *x,
            LayerBias::OnePerNeuron(vec) => vec.iter_mut().for_each(|x| *x *= *x),
        }
        self
    }

    pub fn sqrt_entries_mut(&mut self) -> &mut LayerBias {
        match self {
            LayerBias::OnePerLayer(x) => *x = x.sqrt(),
            LayerBias::OnePerNeuron(vec) => vec.iter_mut().for_each(|x| *x = x.sqrt()),
        }
        self
    }
}

impl AddBias for LayerBias {
    fn add_bias_mut(&mut self, other_bias: &LayerBias) -> &mut Self {
        let biases = (self, other_bias);
        match biases {
            (LayerBias::OnePerLayer(x), LayerBias::OnePerLayer(y)) => *x += y,
            (LayerBias::OnePerNeuron(a), LayerBias::OnePerNeuron(b)) => {
                a.add_entries_mut(b);
            }
            _ => panic!("Cannot add different LayerBias Variants"),
        }
        biases.0
    }
}

impl Randomize for LayerBias {
    type Sample = f64;

    fn _randomize_mut(
        &mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) {
        match self {
            LayerBias::OnePerLayer(x) => *x = rng.sample(distr),
            LayerBias::OnePerNeuron(vec) => vec._randomize_mut(rng, distr),
        }
    }
}

macro_rules! impl_entrywise_arithmetic {
    ( $trait:ident : $fn:ident $op:tt ) => {
        impl $trait<&LayerBias> for LayerBias {
            fn $fn(&mut self, rhs: &LayerBias) -> &mut Self {
                use LayerBias::*;
                let pair = (self, rhs);
                match pair {
                    (OnePerLayer(x), OnePerLayer(y)) =>  *x $op y,
                    (OnePerNeuron(vec1), OnePerNeuron(vec2)) => {
                        vec1.$fn(vec2);
                    }
                    _ => panic!("Cannot add different LayerBias Variants"),
                }
                pair.0
            }
        }
    };
}

impl_entrywise_arithmetic! { EntryAdd: add_entries_mut += }
impl_entrywise_arithmetic! { EntrySub: sub_entries_mut -= }
impl_entrywise_arithmetic! { EntryMul: mul_entries_mut *= }
impl_entrywise_arithmetic! { EntryDiv: div_entries_mut /= }

macro_rules! impl_scalar_arithmetic {
    ( $trait:ident : $fn:ident ) => {
        impl $trait for LayerBias {
            fn $fn(&mut self, scalar: f64) -> &mut Self {
                match self {
                    LayerBias::OnePerLayer(x) => {
                        x.$fn(scalar);
                    }
                    LayerBias::OnePerNeuron(vec) => {
                        vec.$fn(scalar);
                    }
                }
                self
            }
        }
    };
}

impl_scalar_arithmetic! { ScalarAdd : add_scalar_mut }
impl_scalar_arithmetic! { ScalarMul : mul_scalar_mut }
impl_scalar_arithmetic! { ScalarDiv : div_scalar_mut }

impl Lerp<&LayerBias> for LayerBias {
    fn lerp_mut(&mut self, other: &LayerBias, blend: f64) -> &mut Self {
        let biases = (self, other);
        match biases {
            (LayerBias::OnePerLayer(x), LayerBias::OnePerLayer(y)) => {
                x.lerp_mut(y, blend);
            }
            (LayerBias::OnePerNeuron(vec1), LayerBias::OnePerNeuron(vec2)) => {
                vec1.lerp_mut(vec2, blend);
            }
            _ => panic!("Cannot add different LayerBias Variants"),
        }
        biases.0
    }
}

impl std::fmt::Display for LayerBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerBias::OnePerLayer(bias) => write!(f, "{}", bias),
            LayerBias::OnePerNeuron(bias) => write!(f, "{:?}", bias),
        }
    }
}
