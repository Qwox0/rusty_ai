/*
#[allow(unused)]
use super::Pair;
use super::PairList;
use matrix::Element;

/// A list of mathematical vectors with dimension `DIM`.
#[derive(Debug, Clone, derive_more::From)]
pub struct ValueList<X, const DIM: usize>(pub Vec<[X; DIM]>);

impl<X: Element, const DIM: usize> ValueList<X, DIM> {
    /// Uses a function to create a [`Pair`] for every element in `self`.
    pub fn gen_pairs<EO>(self, gen_output: impl Fn([X; DIM]) -> EO) -> PairList<X, DIM, EO> {
        self.0.into_iter().map(|input| (input, gen_output(input))).collect()
    }
}
*/
