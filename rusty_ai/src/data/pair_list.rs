use crate::data::Pair;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::ops::{Deref, Index};

/// implements [`Index<usize, Output = Pair<IN, EO>>`].
#[derive(Debug, Clone, derive_more::From)]
pub struct PairList<const IN: usize, EO>(pub Vec<Pair<IN, EO>>);
/*
pub struct PairList<const IN: usize, EO> {
    inputs: Vec<[f64; IN]>,
    expected_outputs: Vec<EO>,
}
*/

impl<const IN: usize, EO, P> FromIterator<P> for PairList<IN, EO>
where P: Into<Pair<IN, EO>>
{
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        iter.into_iter().map(P::into).collect::<Vec<_>>().into()
    }
}

impl<const IN: usize, EO> PairList<IN, EO> {
    /// if `inputs` and `expected_outputs` have different lengths, the additional elements of the
    /// longer iterator will be ignored.
    pub fn new(
        inputs: impl IntoIterator<Item = [f64; IN]>,
        expected_outputs: impl IntoIterator<Item = impl Into<EO>>,
    ) -> Self {
        let expected_outputs = expected_outputs.into_iter().map(Into::into);
        inputs.into_iter().zip(expected_outputs).collect()
    }

    pub fn with_fn(
        inputs: impl IntoIterator<Item = [f64; IN]>,
        f: impl Fn([f64; IN]) -> EO,
    ) -> Self {
        inputs.into_iter().map(|i| (i, f(i))).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Pair<IN, EO>> {
        self.0.iter()
    }

    pub fn into_iter_tuple(self) -> impl Iterator<Item = ([f64; IN], EO)> {
        self.into_iter().map(Into::into)
    }

    pub fn shuffle_rng(&mut self, rng: &mut impl rand::Rng) {
        self.0.shuffle(rng);
    }

    pub fn shuffle_seeded(&mut self, seed: u64) {
        self.shuffle_rng(&mut StdRng::seed_from_u64(seed));
    }

    pub fn shuffle(&mut self) {
        self.shuffle_rng(&mut rand::thread_rng());
    }

    pub fn choose_multiple<R>(
        &self,
        rng: &mut R,
        amount: usize,
    ) -> impl Iterator<Item = &Pair<IN, EO>>
    where
        R: Rng + ?Sized,
    {
        self.0.choose_multiple(rng, amount)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[Pair<IN, EO>] {
        self.0.as_slice()
    }
}

impl<const IN: usize, EO> Deref for PairList<IN, EO> {
    type Target = [Pair<IN, EO>];

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<const IN: usize, EO> IntoIterator for PairList<IN, EO> {
    type IntoIter = std::vec::IntoIter<Self::Item>;
    type Item = Pair<IN, EO>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const IN: usize, EO> Index<usize> for PairList<IN, EO> {
    type Output = Pair<IN, EO>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// Simple (IN == EO == 1)

impl PairList<1, f64> {
    pub fn from_simple_vecs(vec_in: Vec<f64>, vec_out: Vec<f64>) -> Self {
        vec_in.into_iter().zip(vec_out).collect()
    }

    pub fn into_iter_tuple_simple(self) -> impl Iterator<Item = (f64, f64)> {
        self.into_iter().map(Into::into)
    }

    pub fn iter_tuple_simple(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.iter().map(Pair::simple_tuple)
    }
}
