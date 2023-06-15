use super::{Pair, PairList};

#[derive(Debug, Clone)]
pub struct ValueList<const DIM: usize>(pub Vec<[f64; DIM]>);

impl<const DIM: usize> From<Vec<[f64; DIM]>> for ValueList<DIM> {
    fn from(values: Vec<[f64; DIM]>) -> Self {
        ValueList(values)
    }
}

impl<const DIM: usize> ValueList<DIM> {
    pub fn gen_pairs<const OUT: usize>(
        self,
        gen_output: impl Fn([f64; DIM]) -> [f64; OUT],
    ) -> PairList<DIM, OUT> {
        self.0
            .into_iter()
            .map(|input| Pair::gen(input, &gen_output))
            .collect::<Vec<_>>()
            .into()
    }
}
