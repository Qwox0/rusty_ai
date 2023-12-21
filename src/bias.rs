//! # Bias module

#[allow(unused_imports)]
use crate::layer::Layer;
use crate::traits::ParamsIter;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

/// The bias parameters of a [`Layer`]. This is just a wrapper containing a [`Vec<f64>`].
///
/// Because each neuron in a layer has its own bias, this contains the same number of elements as
/// there are neurons in a layer.
#[derive(
    Debug, Clone, derive_more::From, PartialEq, Serialize, Deserialize, derive_more::IntoIterator,
)]
pub struct LayerBias(#[into_iterator(ref, ref_mut)] Vec<f64>);

impl LayerBias {
    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter(count: usize, iter: impl Iterator<Item = f64>) -> LayerBias {
        let vec: Vec<_> = iter.take(count).collect();
        assert_eq!(vec.len(), count);
        LayerBias(vec)
    }

    /// Sets every bias element in `self` to `value`.
    pub fn fill(&mut self, value: f64) {
        self.0.fill(value);
    }

    /// Creates a [`LayerBias`] that matches the dimensions of `self`, but contains only zeros.
    pub fn clone_with_zeros(&self) -> LayerBias {
        LayerBias(vec![0.0; self.0.len()])
    }

    /// Get the number of biases which is the same as there are neurons in the [`Layer`].
    pub fn get_neuron_count(&self) -> usize {
        self.0.len()
    }
}

impl Deref for LayerBias {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ParamsIter for LayerBias {
    #[allow(refining_impl_trait)]
    fn iter<'a>(&'a self) -> std::slice::Iter<'_, f64> {
        self.into_iter()
    }

    #[allow(refining_impl_trait)]
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, f64> {
        self.into_iter()
    }
}

impl std::fmt::Display for LayerBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
