pub mod aliases;
pub mod layer;

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::{Flatten, Map},
    slice::{Iter, IterMut},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, derive_more::From)]
pub struct Gradient {
    pub(crate) layers: Vec<GradientLayer>,
}

impl Gradient {
    pub fn set_zero(&mut self) {
        for l in self.layers.iter_mut() {
            l.bias_gradient.fill(0.0);
            l.weight_gradient.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn iter_<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a f64> {
        self.layers.iter().map(GradientLayer::iter).flatten()
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


impl std::fmt::Display for Gradient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self
            .layers
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "{}", text)
    }
}
