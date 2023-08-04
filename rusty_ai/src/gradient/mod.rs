pub mod aliases;
pub mod layer;

use crate::prelude::*;

#[derive(Debug, Clone, derive_more::From)]
pub struct Gradient {
    layers: Vec<GradientLayer>,
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

impl IterLayerParams for Gradient {
    type Layer = GradientLayer;

    fn iter_layers<'a>(&'a self) -> impl Iterator<Item = &'a Self::Layer> {
        self.layers.iter()
    }

    fn iter_mut_layers<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Layer> {
        self.layers.iter_mut()
    }
}

impl std::fmt::Display for Gradient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self.layers.iter().map(ToString::to_string).collect::<Vec<String>>().join("\n");
        write!(f, "{}", text)
    }
}
