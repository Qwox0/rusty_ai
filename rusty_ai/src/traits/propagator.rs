use crate::prelude::*;

pub trait Propagator<const IN: usize, const OUT: usize> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT>;

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT>;
}
