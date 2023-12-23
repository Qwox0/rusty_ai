use crate::data::value_list::ValueList;
use rand::{
    distributions::Uniform,
    prelude::Distribution,
    rngs::{StdRng, ThreadRng},
    Rng, SeedableRng,
};

/// Build a [`ValueList`] with random values. The [`Distribution`] `D` is used for the generation.
///
/// uses [`rand::thread_rng`] by default.
pub struct DataBuilder<D: Distribution<f64>, RNG: rand::Rng> {
    distr: D,
    rng: RNG,
}

impl Default for DataBuilder<Uniform<f64>, ThreadRng> {
    /// Creates values uniformly between `0.0` and `1.0`.
    fn default() -> Self {
        DataBuilder { distr: Uniform::from(0.0..1.0), rng: rand::thread_rng() }
    }
}

impl DataBuilder<Uniform<f64>, ThreadRng> {
    /// Creates a [`DataBuilder`] with a [`Uniform`] distributions in the range `range`.
    pub fn uniform(range: impl Into<Uniform<f64>>) -> DataBuilder<Uniform<f64>, ThreadRng> {
        DataBuilder::with_distr(range.into())
    }
}

impl<D: Distribution<f64>> DataBuilder<D, ThreadRng> {
    /// Creates a [`DataBuilder`] with the distribution `distr`.
    pub fn with_distr(distr: D) -> DataBuilder<D, ThreadRng> {
        DataBuilder { distr, ..Default::default() }
    }
}

impl<D: Distribution<f64>, RNG: rand::Rng> DataBuilder<D, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> DataBuilder<D, R> {
        DataBuilder { rng, ..self }
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> DataBuilder<D, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }
}

impl<D: Distribution<f64>, RNG: rand::Rng> DataBuilder<D, RNG> {
    /// Sets the `distr`.
    pub fn distr<ND: Distribution<f64>>(self, distr: ND) -> DataBuilder<ND, RNG> {
        DataBuilder { distr, ..self }
    }

    /// Generate a [`ValueList`] with `data_count` elements where every element contains `N`
    /// numbers [`f64`].
    pub fn build<const N: usize>(&mut self, data_count: usize) -> ValueList<N> {
        (&mut self.rng)
            .sample_iter(&self.distr)
            .array_chunks()
            .take(data_count)
            .collect::<Vec<_>>()
            .into()
    }
}
