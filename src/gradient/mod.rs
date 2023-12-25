//! # Gradient module
//!
//! This module contains type definitions for documentation purposes.

#[allow(unused_imports)]
use crate::NeuralNetwork;
use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

pub mod aliases;
mod layer;
pub use self::layer::GradientLayer;

/// Contains the derivatives of the total loss with respect to the parameters of a [`NeuralNetwork`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, derive_more::From)]
pub struct Gradient {
    pub(crate) layers: Vec<GradientLayer>,
}

impl Gradient {
    /// Sets every element of `self` to `0`.
    pub fn set_zero(&mut self) {
        for l in self.layers.iter_mut() {
            l.bias_gradient.fill(0.0);
            l.weight_gradient.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn _add(&mut self, rhs: &Gradient) {
        for (r, l) in self.iter_mut().zip(rhs.iter()) {
            *r += *l;
        }
    }
}

impl FromIterator<GradientLayer> for Gradient {
    fn from_iter<T: IntoIterator<Item = GradientLayer>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl ParamsIter for Gradient {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a f64> {
        self.layers.iter().map(GradientLayer::iter).flatten()
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut f64> {
        self.layers.iter_mut().map(GradientLayer::iter_mut).flatten()
    }
}

impl Add<&Self> for Gradient {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self._add(rhs);
        self
    }
}

impl Add<Self> for Gradient {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl AddAssign<&Self> for Gradient {
    fn add_assign(&mut self, rhs: &Self) {
        self._add(rhs)
    }
}

impl AddAssign<Self> for Gradient {
    fn add_assign(&mut self, rhs: Self) {
        self._add(&rhs)
    }
}

impl std::fmt::Display for Gradient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", text)
    }
}
