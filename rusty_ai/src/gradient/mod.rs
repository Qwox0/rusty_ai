pub mod aliases;
pub mod layer;

use self::layer::GradientLayer;
use crate::{
    traits::IterLayerParams,
    util::{EntryAdd, ScalarDiv, ScalarMul},
};

#[derive(Debug, Clone)]
pub struct Gradient {
    layers: Vec<GradientLayer>,
}

impl From<Vec<GradientLayer>> for Gradient {
    fn from(layers: Vec<GradientLayer>) -> Self {
        Gradient { layers }
    }
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
            l.bias_gradient.fill_mut(0.0);
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

impl EntryAdd for Gradient {
    fn add_entries_mut(&mut self, rhs: Self) -> &mut Self {
        self.layers.add_entries_mut(rhs.layers);
        self
    }
}

impl ScalarMul for Gradient {
    fn mul_scalar_mut(&mut self, scalar: f64) -> &mut Self {
        self.layers.mul_scalar_mut(scalar);
        self
    }
}

impl ScalarDiv for Gradient {
    fn div_scalar_mut(&mut self, scalar: f64) -> &mut Self {
        self.layers.div_scalar_mut(scalar);
        self
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
