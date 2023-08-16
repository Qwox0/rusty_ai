use crate::data::Pair;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::ops::{Deref, Index};

#[derive(Debug, thiserror::Error)]
pub enum E {
    #[error("Lengths are not equal.")]
    UnequalLen,
}

/// implements [`Index<usize, Output = Pair<IN, EO>>`].
#[derive(Debug, Clone, derive_more::From)]
pub struct PairList<const IN: usize, EO>(pub Vec<Pair<IN, EO>>);

impl<const IN: usize, EO, P> FromIterator<P> for PairList<IN, EO>
where P: Into<Pair<IN, EO>>
{
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        iter.into_iter().map(P::into).collect::<Vec<_>>().into()
    }
}

impl<const IN: usize, EO> PairList<IN, EO> {
    pub fn with_fn(
        inputs: impl Into<Vec<[f64; IN]>>,
        f: impl Fn([f64; IN]) -> EO,
    ) -> Result<Self, E> {
        let inputs = inputs.into();
        let outputs: Vec<_> = inputs.iter().map(Clone::clone).map(f).collect();
        PairList::from_vecs(inputs, outputs)
    }

    pub fn from_vecs(
        vec_in: impl Into<Vec<[f64; IN]>>,
        vec_out: impl Into<Vec<EO>>,
    ) -> Result<Self, E> {
        let vec_in = vec_in.into();
        let vec_out = vec_out.into();
        if vec_in.len() != vec_out.len() {
            Err(E::UnequalLen)
        } else {
            Ok(vec_in.into_iter().zip(vec_out).collect())
        }
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
