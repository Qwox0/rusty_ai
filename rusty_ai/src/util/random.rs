use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    rngs::StdRng,
    Rng, SeedableRng,
};
use std::{ops::Range, usize};

pub trait Randomize: Sized {
    type Sample;

    fn _randomize_mut(
        &mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    );

    fn _randomize(
        mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) -> Self {
        self._randomize_mut(rng, distr);
        self
    }

    fn randomize<D: Distribution<Self::Sample>>(self, distr: D) -> Self {
        self._randomize(&mut rand::thread_rng(), distr)
    }

    fn randomize_seeded<D: Distribution<Self::Sample>>(self, seed: u64, distr: D) -> Self {
        self._randomize(&mut StdRng::seed_from_u64(seed), distr)
    }

    fn _randomize_uniform(self, rng: &mut impl Rng, range: Range<Self::Sample>) -> Self
    where
        Self::Sample: SampleUniform,
    {
        self._randomize(rng, Uniform::from(range))
    }

    fn randomize_uniform(self, range: Range<Self::Sample>) -> Self
    where
        Self::Sample: SampleUniform,
    {
        self._randomize_uniform(&mut rand::thread_rng(), range)
    }

    fn randomize_uniform_seeded(self, range: Range<Self::Sample>, seed: u64) -> Self
    where
        Self::Sample: SampleUniform,
    {
        self._randomize_uniform(&mut StdRng::seed_from_u64(seed), range)
    }
}

impl<const N: usize> Randomize for [f64; N] {
    type Sample = f64;

    fn _randomize_mut(
        &mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) {
        self.iter_mut().for_each(|x| *x = rng.sample(&distr));
    }
}

impl<T> Randomize for Vec<T> {
    type Sample = T;

    fn _randomize_mut(
        &mut self,
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) {
        self.iter_mut().for_each(|x| *x = rng.sample(&distr));
    }
}

/// create a randomized self with samples of type `Sample`
pub trait Random: Sized {
    type Sample;

    fn _random(
        rng: &mut impl rand::Rng,
        distr: impl rand::distributions::Distribution<Self::Sample>,
    ) -> Self;

    fn random<D: Distribution<Self::Sample>>(distr: D) -> Self {
        Self::_random(&mut rand::thread_rng(), distr)
    }

    fn random_seeded<D: Distribution<Self::Sample>>(seed: u64, distr: D) -> Self {
        Self::_random(&mut StdRng::seed_from_u64(seed), distr)
    }

    // uniform
    fn _random_uniform(rng: &mut impl Rng, range: Range<Self::Sample>) -> Self
    where
        Self::Sample: SampleUniform,
    {
        Self::_random(rng, Uniform::from(range))
    }

    fn random_uniform(range: Range<Self::Sample>) -> Self
    where
        Self::Sample: SampleUniform,
    {
        Self::_random_uniform(&mut rand::thread_rng(), range)
    }

    fn random_uniform_seeded(range: Range<Self::Sample>, seed: u64) -> Self
    where
        Self::Sample: SampleUniform,
    {
        Self::_random_uniform(&mut StdRng::seed_from_u64(seed), range)
    }
}

impl Random for f64 {
    type Sample = f64;

    fn _random(rng: &mut impl Rng, distr: impl rand::distributions::Distribution<f64>) -> Self {
        rng.sample(distr)
    }

    fn random_uniform(range: Range<Self::Sample>) -> Self
    where
        Self::Sample: SampleUniform,
    {
        rand::thread_rng().gen_range(range)
    }
}

impl<const N: usize> Random for [f64; N] {
    type Sample = f64;

    fn _random(rng: &mut impl Rng, distr: impl rand::distributions::Distribution<f64>) -> Self {
        rng.sample_iter(distr).array_chunks().next().unwrap()
    }
}

pub trait MultiRandom: Sized {
    type Sample;
    type Size;

    fn _random_multiple(
        rng: &mut impl Rng,
        distr: impl Distribution<Self::Sample>,
        count: Self::Size,
    ) -> Self;

    fn random_multiple(distr: impl Distribution<Self::Sample>, count: Self::Size) -> Self {
        Self::_random_multiple(&mut rand::thread_rng(), distr, count)
    }

    fn random_uniform_multiple(range: Range<Self::Sample>, count: Self::Size) -> Self
    where
        Self::Sample: SampleUniform,
    {
        Self::_random_multiple(&mut rand::thread_rng(), Uniform::from(range), count)
    }

    fn random_uniform_seeded_multiple(
        range: Range<Self::Sample>,
        seed: u64,
        count: Self::Size,
    ) -> Self
    where
        Self::Sample: SampleUniform,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::_random_multiple(&mut rng, Uniform::from(range), count)
    }
}
