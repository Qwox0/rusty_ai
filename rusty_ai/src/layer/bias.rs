use std::marker::PhantomData;

use super::AddBias;
use crate::{
    gradient::aliases::{BiasGradient, WeightedSumGradient},
    util::{EntryAdd, EntryDiv, EntryMul, EntrySub, Lerp, ScalarAdd, ScalarDiv, ScalarMul},
};
use rand::{prelude::Distribution, Rng};

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

    pub fn fill_mut(&mut self, value: f64) {
        match self {
            LayerBias::OnePerLayer(x) => *x = value,
            LayerBias::OnePerNeuron(vec) => vec.fill(value),
        }
    }

    pub fn fill(mut self, value: f64) -> LayerBias {
        self.fill_mut(value);
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

    pub fn iter<'a>(&'a self) -> IterLayerBias<'a> {
        IterLayerBias::new(self)
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMutLayerBias<'a> {
        IterMutLayerBias::new(self)
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

#[derive(Clone)]
pub struct IterLayerBias<'a> {
    bias: &'a LayerBias,
    idx: usize,
    len: usize,
}

impl<'a> IterLayerBias<'a> {
    pub fn new(bias: &'a LayerBias) -> Self {
        let len = bias.get_neuron_count().unwrap_or(1);
        IterLayerBias { bias, idx: 0, len }
    }
}

impl<'a> Iterator for IterLayerBias<'a> {
    type Item = &'a f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.len {
            return None;
        }
        let idx = self.idx;
        self.idx += 1;
        match self.bias {
            LayerBias::OnePerLayer(f) => Some(f),
            LayerBias::OnePerNeuron(v) => v.get(idx),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len;
        (exact, Some(exact))
    }
}

impl<'a> ExactSizeIterator for IterLayerBias<'a> {}

pub struct IterMutLayerBias<'a> {
    ptr: *mut f64,
    end: *mut f64,
    _marker: PhantomData<&'a mut LayerBias>,
}

impl<'a> IterMutLayerBias<'a> {
    pub fn new(bias: &'a mut LayerBias) -> Self {
        let (ptr, len) = match bias {
            LayerBias::OnePerLayer(f) => (f as *mut f64, 1),
            LayerBias::OnePerNeuron(v) => (v.as_mut_ptr(), v.len()),
        };
        let end = unsafe { ptr.add(len) };
        let _marker = PhantomData;
        IterMutLayerBias { ptr, end, _marker }
    }
}

impl<'a> Iterator for IterMutLayerBias<'a> {
    type Item = &'a mut f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            return None;
        }
        let val = unsafe { &mut *self.ptr };
        unsafe { self.ptr = self.ptr.add(1) };
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = unsafe { self.end.sub_ptr(self.ptr) };
        (exact, Some(exact))
    }
}

impl<'a> ExactSizeIterator for IterMutLayerBias<'a> {}
