use crate::{
    gradient::aliases::{BiasGradient, WeightedSumGradient},
    util::{EntryAdd, EntryDiv, EntryMul, EntrySub, Lerp, ScalarAdd, ScalarDiv, ScalarMul},
};

#[derive(Debug, Clone)]
pub struct LayerBias(Vec<f64>);

impl LayerBias {
    pub fn new(bias: Vec<f64>) -> LayerBias {
        LayerBias(bias)
    }

    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter(count: usize, iter: impl Iterator<Item = f64>) -> LayerBias {
        let vec: Vec<_> = iter.take(count).collect();
        assert_eq!(vec.len(), count);
        LayerBias::new(vec)
    }

    pub fn fill_mut(&mut self, value: f64) {
        self.0.fill(value);
    }

    pub fn clone_with_zeros(&self) -> LayerBias {
        LayerBias(vec![0.0; self.0.len()])
    }

    pub fn new_matching_gradient(
        &self,
        weighted_sum_gradient: &WeightedSumGradient,
    ) -> BiasGradient {
        LayerBias(weighted_sum_gradient.clone())
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

    pub fn iter(&self) -> core::slice::Iter<'_, f64> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, f64> {
        self.0.iter_mut()
    }
}

macro_rules! impl_entrywise_arithmetic {
    ( $trait:ident : $fn:ident ) => {
        impl $trait<&LayerBias> for LayerBias {
            fn $fn(&mut self, rhs: &LayerBias) -> &mut Self {
                self.0.$fn(rhs.get_vec());
                self
            }
        }
    };
}

impl_entrywise_arithmetic! { EntryAdd: add_entries_mut }
impl_entrywise_arithmetic! { EntrySub: sub_entries_mut }
impl_entrywise_arithmetic! { EntryMul: mul_entries_mut }
impl_entrywise_arithmetic! { EntryDiv: div_entries_mut }

macro_rules! impl_scalar_arithmetic {
    ( $trait:ident : $fn:ident ) => {
        impl $trait for LayerBias {
            fn $fn(&mut self, scalar: f64) -> &mut Self {
                self.0.$fn(scalar);
                self
            }
        }
    };
}

impl_scalar_arithmetic! { ScalarAdd : add_scalar_mut }
impl_scalar_arithmetic! { ScalarMul : mul_scalar_mut }
impl_scalar_arithmetic! { ScalarDiv : div_scalar_mut }

impl Lerp<&LayerBias> for LayerBias {
    fn lerp_mut(&mut self, other: &LayerBias, blend: f64) -> &mut Self {
        self.0.lerp_mut(other.get_vec(), blend);
        self
    }
}

impl std::fmt::Display for LayerBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
