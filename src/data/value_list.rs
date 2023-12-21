#[allow(unused)]
use super::Pair;
use super::PairList;

/// A list of mathematical vectors with dimension `DIM`.
#[derive(Debug, Clone, derive_more::From)]
pub struct ValueList<const DIM: usize>(pub Vec<[f64; DIM]>);

impl<const DIM: usize> ValueList<DIM> {
    /// Uses a function to create a [`Pair`] for every element in `self`.
    pub fn gen_pairs<EO>(self, gen_output: impl Fn([f64; DIM]) -> EO) -> PairList<DIM, EO> {
        self.0.into_iter().map(|input| (input, gen_output(input))).collect()
    }
}
