pub mod aliases;
pub mod layer;

use crate::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, derive_more::From)]
pub struct Gradient {
    pub(crate) layers: Vec<GradientLayer>,
}

impl Gradient {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn iter_mut_layers(&mut self) -> core::slice::IterMut<'_, GradientLayer> {
        self.layers.iter_mut()
    }

    pub fn set_zero(&mut self) {
        for l in self.layers.iter_mut() {
            l.bias_gradient.fill(0.0);
            l.weight_gradient.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

impl FromIterator<GradientLayer> for Gradient {
    fn from_iter<T: IntoIterator<Item = GradientLayer>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<'a> LayerIter<'a> for Gradient {
    type Iter = core::slice::Iter<'a, Self::Layer>;
    type IterMut = core::slice::IterMut<'a, Self::Layer>;
    type Layer = GradientLayer;

    fn iter_layers(&'a self) -> Self::Iter {
        self.layers.iter()
    }

    fn iter_mut_layers(&'a mut self) -> Self::IterMut {
        self.layers.iter_mut()
    }
}

impl std::fmt::Display for Gradient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", text)
    }
}
