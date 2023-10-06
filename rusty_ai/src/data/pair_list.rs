use crate::input::Input;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

pub type Pair<const IN: usize, EO> = (Input<IN>, EO);

/// implements [`Index<usize, Output = Pair<IN, EO>>`].
#[derive(
    Debug,
    Clone,
    derive_more::From,
    derive_more::Deref,
    derive_more::IntoIterator,
    derive_more::Index,
)]
pub struct PairList<const IN: usize, EO>(pub Vec<Pair<IN, EO>>);

impl<const IN: usize, I, EO> FromIterator<(I, EO)> for PairList<IN, EO>
where I: Into<Input<IN>>
{
    fn from_iter<T: IntoIterator<Item = (I, EO)>>(iter: T) -> Self {
        iter.into_iter()
            .map(|(i, eo)| (i.into(), eo))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<const IN: usize, EO> PairList<IN, EO> {
    /// if `inputs` and `expected_outputs` have different lengths, the additional elements of the
    /// longer iterator will be ignored.
    pub fn new(
        inputs: impl IntoIterator<Item = Input<IN>>,
        expected_outputs: impl IntoIterator<Item = impl Into<EO>>,
    ) -> Self {
        let expected_outputs = expected_outputs.into_iter().map(Into::into);
        inputs.into_iter().zip(expected_outputs).collect()
    }

    pub fn with_fn(
        inputs: impl IntoIterator<Item = Input<IN>>,
        f: impl Fn(&Input<IN>) -> EO,
    ) -> Self {
        inputs
            .into_iter()
            .map(|i| {
                let eo = f(&i);
                (i, eo)
            })
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(Input<IN>, EO)> {
        self.0.iter()
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

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[(Input<IN>, EO)] {
        self.0.as_slice()
    }
}

impl<'a, const IN: usize, EO> IntoIterator for &'a PairList<IN, EO> {
    type IntoIter = std::slice::Iter<'a, (Input<IN>, EO)>;
    type Item = &'a (Input<IN>, EO);

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl PairList<1, f64> {
    pub fn from_simple_vecs(vec_in: Vec<f64>, vec_out: Vec<f64>) -> Self {
        vec_in
            .into_iter()
            .map(|x| Input::from([x]))
            .zip(vec_out)
            .collect()
    }
}
