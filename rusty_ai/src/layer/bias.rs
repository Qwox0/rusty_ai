use serde::{Deserialize, Serialize};

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

    pub fn fill(&mut self, value: f64) {
        self.0.fill(value);
    }

    pub fn clone_with_zeros(&self) -> LayerBias {
        LayerBias(vec![0.0; self.0.len()])
    }

    pub fn get_vec(&self) -> &Vec<f64> {
        &self.0
    }

    pub fn get_neuron_count(&self) -> usize {
        self.0.len()
    }

    pub fn sqare_entries_mut(&mut self) -> &mut LayerBias {
        self.0.iter_mut().for_each(|x| *x *= *x);
        self
    }

    pub fn sqrt_entries_mut(&mut self) -> &mut LayerBias {
        self.0.iter_mut().for_each(|x| *x = x.sqrt());
        self
    }

    pub fn iter<'a>(&'a self) -> core::slice::Iter<'a, f64> {
        self.0.iter()
    }
}

impl std::fmt::Display for LayerBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
