use crate::{data::value_list::ValueList, util::RngWrapper};
use rand::{distributions::Uniform, prelude::Distribution, Rng};

/// Build a [`ValueList`] with random values. The [`Distribution`] `D` is used for the generation.
pub struct DataBuilder<D: Distribution<f64>> {
    distr: D,
    rng_seed: Option<u64>,
}

impl Default for DataBuilder<Uniform<f64>> {
    /// Creates values uniformly between `0` and `1`.
    fn default() -> Self {
        DataBuilder { distr: Uniform::from(0.0..1.0), rng_seed: None }
    }
}

impl DataBuilder<Uniform<f64>> {
    /// Creates a [`DataBuilder`] with a [`Uniform`] distributions in the range `range`.
    pub fn uniform(range: impl Into<Uniform<f64>>) -> DataBuilder<Uniform<f64>> {
        DataBuilder::with_distr(range.into())
    }
}

impl<D: Distribution<f64>> DataBuilder<D> {
    /// Creates a [`DataBuilder`] with the distribution `distr`.
    pub fn with_distr(distr: D) -> DataBuilder<D> {
        DataBuilder { distr, ..Default::default() }
    }

    /// Sets the `distr`.
    pub fn distr<ND: Distribution<f64>>(self, distr: ND) -> DataBuilder<ND> {
        DataBuilder { distr, ..self }
    }

    /// Sets the `seed`.
    pub fn seed(mut self, seed: u64) -> Self {
        let _ = self.rng_seed.insert(seed);
        self
    }

    /// Generate a [`ValueList`] with `data_count` elements where every element contains `N`
    /// numbers [`f64`].
    pub fn build<const N: usize>(&self, data_count: usize) -> ValueList<N> {
        RngWrapper::new(self.rng_seed)
            .sample_iter(&self.distr)
            .array_chunks()
            .take(data_count)
            .collect::<Vec<_>>()
            .into()
    }
}
