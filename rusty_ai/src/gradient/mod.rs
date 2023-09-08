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

impl<'a> ParamsIter<'a> for Gradient {
    type Iter = Flatten<
        Map<
            Iter<'a, GradientLayer>,
            impl FnMut(&'a GradientLayer) -> <GradientLayer as ParamsIter>::Iter,
        >,
    >;
    type IterMut = Flatten<
        Map<
            IterMut<'a, GradientLayer>,
            impl FnMut(&'a mut GradientLayer) -> <GradientLayer as ParamsIter>::IterMut,
        >,
    >;

    fn iter(&'a self) -> Self::Iter {
        self.layers.iter().map(GradientLayer::iter).flatten()
    }

    fn iter_mut(&'a mut self) -> Self::IterMut {
        self.layers
            .iter_mut()
            .map(GradientLayer::iter_mut)
            .flatten()
    }
}

impl_IntoIterator! { Gradient}

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
