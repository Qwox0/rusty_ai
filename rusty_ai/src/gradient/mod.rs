pub mod aliases;
pub mod layer;

use self::layer::GradientLayer;
use crate::util::{EntryAdd, ScalarDiv, ScalarMul};

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

    pub fn iter_layers(&self) -> core::slice::Iter<'_, GradientLayer> {
        self.layers.iter()
    }

    pub fn iter_mut_layers(&mut self) -> core::slice::IterMut<'_, GradientLayer> {
        self.layers.iter_mut()
    }

    pub fn iter_numbers(&self) -> impl Iterator<Item = &f64> {
        self.layers.iter().map(|l| l.iter_numbers()).flatten()
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

impl IntoIterator for Gradient {
    type Item = GradientLayer;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}
