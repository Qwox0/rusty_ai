use crate::propagation::PropagationResult;

#[derive(Debug)]
pub struct TestsResult<const OUT: usize> {
    pub outputs: Vec<PropagationResult<OUT>>,
    pub error: f64,
}

impl<const OUT: usize> FromIterator<(PropagationResult<OUT>, f64)> for TestsResult<OUT> {
    fn from_iter<T: IntoIterator<Item = (PropagationResult<OUT>, f64)>>(iter: T) -> Self {
        let (outputs, errors): (_, Vec<_>) = iter.into_iter().unzip();
        let error = errors.into_iter().sum();
        TestsResult { outputs, error }
    }
}

impl<const OUT: usize> TestsResult<OUT> {
    pub fn collect(iter: impl Iterator<Item = (PropagationResult<OUT>, f64)>) -> Self {
        let (outputs, errors): (_, Vec<_>) = iter.unzip();
        let error = errors.into_iter().sum();
        /*
        let (outputs, error) = iter.fold((vec![], 0.0), |mut acc, (out, err)| {
            acc.0.push(out);
            acc.1 += err;
            acc
        });
        */
        TestsResult { outputs, error }
    }
}

pub struct TrainingsResult<'a, const IN: usize, const OUT: usize> {
    pub input: &'a [f64; IN],
    pub output: [f64; OUT],
    pub generation: usize,
}
