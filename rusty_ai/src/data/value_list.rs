use super::{Pair, PairList};

#[derive(Debug, Clone, derive_more::From)]
pub struct ValueList<const DIM: usize>(pub Vec<[f64; DIM]>);

impl<const DIM: usize> ValueList<DIM> {
    pub fn gen_pairs<EO>(self, gen_output: impl Fn([f64; DIM]) -> EO) -> PairList<DIM, EO> {
        self.0.into_iter().map(|input| Pair::with(input, &gen_output)).collect()
    }
}
