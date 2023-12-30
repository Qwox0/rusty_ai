use crate::data::value_list::ValueList;
use matrix::Num;
use rand::{
    distributions::Uniform,
    prelude::Distribution,
    rngs::{StdRng, ThreadRng},
    Rng, SeedableRng,
};
use std::marker::PhantomData;

/// Build a [`ValueList`] with random values. The [`Distribution`] `D` is used for the generation.
///
/// uses [`rand::thread_rng`] by default.
pub struct DataBuilder<X, D: Distribution<X>, RNG: rand::Rng> {
    distr: D,
    rng: RNG,
    _marker: PhantomData<X>,
}

impl<X: Num> Default for DataBuilder<X, Uniform<X>, ThreadRng> {
    /// Creates values uniformly between `0.0` and `1.0`.
    fn default() -> Self {
        DataBuilder {
            distr: Uniform::from(X::lit(0)..X::lit(1)),
            rng: rand::thread_rng(),
            _marker: PhantomData,
        }
    }
}

impl<X: Num> DataBuilder<X, Uniform<X>, ThreadRng> {
    /// Creates a [`DataBuilder`] with a [`Uniform`] distributions in the range `range`.
    pub fn uniform(range: impl Into<Uniform<X>>) -> Self {
        DataBuilder::with_distr(range.into())
    }
}

impl<X: Num, D: Distribution<X>> DataBuilder<X, D, ThreadRng> {
    /// Creates a [`DataBuilder`] with the distribution `distr`.
    pub fn with_distr(distr: D) -> DataBuilder<X, D, ThreadRng> {
        DataBuilder { distr, ..Default::default() }
    }
}

impl<X, D: Distribution<X>, RNG: rand::Rng> DataBuilder<X, D, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> DataBuilder<X, D, R> {
        DataBuilder { rng, ..self }
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> DataBuilder<X, D, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }
}

impl<X, D: Distribution<X>, RNG: rand::Rng> DataBuilder<X, D, RNG> {
    /// Sets the `distr`.
    pub fn distr<ND: Distribution<X>>(self, distr: ND) -> DataBuilder<X, ND, RNG> {
        DataBuilder { distr, ..self }
    }

    /// Generate a [`ValueList`] with `data_count` elements where every element contains `N`
    /// numbers [`X`].
    pub fn build<const N: usize>(&mut self, data_count: usize) -> ValueList<X, N> {
        (&mut self.rng)
            .sample_iter(&self.distr)
            .array_chunks()
            .take(data_count)
            .collect::<Vec<_>>()
            .into()
    }
}
