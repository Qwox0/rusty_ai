/*
//! # Bias module

#[allow(unused_imports)]
use crate::layer::Layer;
use crate::{traits::ParamsIter, Element};
use core::slice;
use matrix::Num;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    ops::Deref,
};

/// The bias parameters of a [`Layer`]. This is just a wrapper containing a [`Vec<f64>`].
///
/// Because each neuron in a layer has its own bias, this contains the same number of elements as
/// there are neurons in a layer.
#[derive(Debug, Clone, derive_more::From, PartialEq, Serialize, Deserialize)]
pub struct LayerBias<X>(Vec<X>);

impl<'a, X> IntoIterator for &'a LayerBias<X> {
    type IntoIter = slice::Iter<'a, X>;
    type Item = &'a X;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, X> IntoIterator for &'a mut LayerBias<X> {
    type IntoIter = slice::IterMut<'a, X>;
    type Item = &'a mut X;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<X> LayerBias<X> {
    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter(count: usize, iter: impl Iterator<Item = X>) -> LayerBias<X> {
        let vec: Vec<_> = iter.take(count).collect();
        assert_eq!(vec.len(), count);
        LayerBias(vec)
    }

    /// Get the number of biases which is the same as there are neurons in the [`Layer`].
    pub fn get_neuron_count(&self) -> usize {
        self.0.len()
    }
}

impl<X: Element> LayerBias<X> {
    /// Sets every bias element in `self` to `value`.
    pub fn fill(&mut self, value: X) {
        self.0.fill(value);
    }
}

impl<X: Num> LayerBias<X> {
    /// Creates a [`LayerBias<X>`] that matches the dimensions of `self`, but contains only zeros.
    pub fn clone_with_zeros(&self) -> LayerBias<X> {
        LayerBias(vec![X::zero(); self.0.len()])
    }
}

impl<X> Deref for LayerBias<X> {
    type Target = [X];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<X: Element> ParamsIter<X> for LayerBias<X> {
    #[allow(refining_impl_trait)]
    fn iter<'a>(&'a self) -> std::slice::Iter<'_, X> {
        self.into_iter()
    }

    #[allow(refining_impl_trait)]
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, X> {
        self.into_iter()
    }
}

impl<X: Debug> Display for LayerBias<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
*/
