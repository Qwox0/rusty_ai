//! # Gradient<X> module
//!
//! This module contains type definitions for documentation purposes.

#[allow(unused_imports)]
use crate::NeuralNetwork;
use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    iter,
    ops::{Add, AddAssign},
};

pub mod aliases;
mod layer;
pub use self::layer::GradientLayer;

/// Contains the derivatives of the total loss with respect to the parameters of a [`NeuralNetwork`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, derive_more::From)]
pub struct Gradient<X> {
    pub(crate) layers: Vec<GradientLayer<X>>,
}

impl<X: Num> Gradient<X> {
    /// Sets every element of `self` to `0`.
    pub fn set_zero(&mut self) {
        for l in self.layers.iter_mut() {
            l.bias_gradient.fill(X::zero());
            l.weight_gradient.iter_mut().for_each(|x| *x = X::zero());
        }
    }

    fn _add(&mut self, rhs: &Gradient<X>) {
        for (r, l) in self.iter_mut().zip(rhs.iter()) {
            *r += *l;
        }
    }
}

impl<X> FromIterator<GradientLayer<X>> for Gradient<X> {
    fn from_iter<T: IntoIterator<Item = GradientLayer<X>>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<X: Element> ParamsIter<X> for Gradient<X> {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a X> {
        self.layers.iter().map(GradientLayer::iter).flatten()
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut X> {
        self.layers.iter_mut().map(GradientLayer::iter_mut).flatten()
    }
}

impl<X: Num> Add<&Self> for Gradient<X> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self._add(rhs);
        self
    }
}

impl<X: Num> Add<Self> for Gradient<X> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<X: Num> AddAssign<&Self> for Gradient<X> {
    fn add_assign(&mut self, rhs: &Self) {
        self._add(rhs)
    }
}

impl<X: Num> AddAssign<Self> for Gradient<X> {
    fn add_assign(&mut self, rhs: Self) {
        self._add(&rhs)
    }
}

impl<X: Element> std::fmt::Display for Gradient<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", text)
    }
}
