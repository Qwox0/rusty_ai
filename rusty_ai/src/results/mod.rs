mod propagation;

pub use propagation::{PropagationResult, VerbosePropagation};

#[derive(Debug)]
pub struct TestsResult<const OUT: usize> {
    pub generation: usize,
    pub outputs: Vec<PropagationResult<OUT>>,
    pub error: f64,
}

impl<const OUT: usize> TestsResult<OUT> {
    pub fn collect(
        iter: impl Iterator<Item = (PropagationResult<OUT>, f64)>,
        generation: usize,
    ) -> Self {
        let (outputs, error) = iter.fold((vec![], 0.0), |mut acc, (out, err)| {
            acc.0.push(out);
            acc.1 += err;
            acc
        });
        TestsResult {
            generation,
            outputs,
            error,
        }
    }
}

pub struct TrainingsResult<'a, const IN: usize, const OUT: usize> {
    pub input: &'a [f64; IN],
    pub output: [f64; OUT],
    pub generation: usize,
}
