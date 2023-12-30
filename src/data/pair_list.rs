use crate::input::Input;
use matrix::Element;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
#[allow(unused_imports)]
use {crate::NeuralNetwork, std::ops::Index};

/// A data pair used for training a [`NeuralNetwork`].
pub type Pair<X, const IN: usize, EO> = (Input<X, IN>, EO);

/// A list of data [`Pair`]s used for training a [`NeuralNetwork`].
///
/// implements [`Index<usize, Output = Pair<IN, EO>>`].
#[derive(Debug, Clone, derive_more::From, derive_more::Deref, derive_more::Index)]
pub struct PairList<X, const IN: usize, EO>(pub Vec<Pair<X, IN, EO>>);

impl<X, const IN: usize, I, EO> FromIterator<(I, EO)> for PairList<X, IN, EO>
where I: Into<Input<X, IN>>
{
    fn from_iter<T: IntoIterator<Item = (I, EO)>>(iter: T) -> Self {
        iter.into_iter().map(|(i, eo)| (i.into(), eo)).collect::<Vec<_>>().into()
    }
}

impl<X, const IN: usize, EO> PairList<X, IN, EO> {
    /// if `inputs` and `expected_outputs` have different lengths, the additional elements of the
    /// longer iterator will be ignored.
    pub fn new(
        inputs: impl IntoIterator<Item = Input<X, IN>>,
        expected_outputs: impl IntoIterator<Item = impl Into<EO>>,
    ) -> Self {
        let expected_outputs = expected_outputs.into_iter().map(Into::into);
        inputs.into_iter().zip(expected_outputs).collect()
    }

    /// Creates a list of pairs from an [`Iterator`] over [`Input`]s and a function which creates
    /// the expected outputs.
    pub fn with_fn(
        inputs: impl IntoIterator<Item = Input<X, IN>>,
        mut f: impl FnMut(&Input<X, IN>) -> EO,
    ) -> Self {
        inputs
            .into_iter()
            .map(|i| {
                let eo = f(&i);
                (i, eo)
            })
            .collect()
    }

    /// Returns an [`Iterator`] over the data pairs.
    pub fn iter(&self) -> std::slice::Iter<'_, Pair<X, IN, EO>> {
        self.0.iter()
    }

    /// Shuffle the list using a [`rand::Rng`].
    pub fn shuffle_rng(&mut self, rng: &mut impl rand::Rng) {
        self.0.shuffle(rng);
    }

    /// Shuffle the list using a seed.
    pub fn shuffle_seeded(&mut self, seed: u64) {
        self.shuffle_rng(&mut StdRng::seed_from_u64(seed));
    }

    /// Shuffle the list.
    pub fn shuffle(&mut self) {
        self.shuffle_rng(&mut rand::thread_rng());
    }

    /// Get the length of the list.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<X, const IN: usize, EO> IntoIterator for PairList<X, IN, EO> {
    type IntoIter = std::vec::IntoIter<Self::Item>;
    type Item = Pair<X, IN, EO>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, X, const IN: usize, EO> IntoIterator for &'a PairList<X, IN, EO> {
    type IntoIter = std::slice::Iter<'a, (Input<X, IN>, EO)>;
    type Item = &'a Pair<X, IN, EO>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<X: Element, const IN: usize, EO: Send> IntoParallelIterator for PairList<X, IN, EO> {
    type Item = Pair<X, IN, EO>;
    type Iter = rayon::vec::IntoIter<Self::Item>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}

impl<'a, X: Element, const IN: usize, EO: Sync> IntoParallelIterator for &'a PairList<X, IN, EO> {
    type Item = &'a Pair<X, IN, EO>;
    type Iter = rayon::slice::Iter<'a, Pair<X, IN, EO>>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl<X> PairList<X, 1, X> {
    /// Create a [`PairList`] for a [`NeuralNetwork`] with 1 input and 1 output
    pub fn from_simple_vecs(vec_in: Vec<X>, vec_out: Vec<X>) -> Self {
        vec_in.into_iter().map(|x| Input::from([x])).zip(vec_out).collect()
    }
}
