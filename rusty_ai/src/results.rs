use crate::util::SetLength;

pub struct PropagationResult<const OUT: usize>(pub [f64; OUT]);

impl<const OUT: usize> From<Vec<f64>> for PropagationResult<OUT> {
    fn from(value: Vec<f64>) -> Self {
        PropagationResult(value.to_arr(f64::default()))
    }
}
impl<const OUT: usize> From<[f64; OUT]> for PropagationResult<OUT> {
    fn from(output: [f64; OUT]) -> Self {
        PropagationResult(output)
    }
}

impl<const OUT: usize> Into<[f64; OUT]> for PropagationResult<OUT> {
    fn into(self) -> [f64; OUT] {
        self.0
    }
}

impl<const OUT: usize> PropagationResult<OUT> {
    /// Mean squarred error: E = 0.5 * ∑ (o_i - t_i)^2 from i = 1 to n
    pub fn mean_squarred_error(&self, expected_output: &[f64; OUT]) -> f64 {
        0.5 * self
            .0
            .iter()
            .zip(expected_output)
            .map(|(out, expected)| out - expected)
            .map(|x| x * x)
            .sum::<f64>()
    }
}

#[derive(Debug)]
pub struct TestsResult<const IN: usize, const OUT: usize> {
    pub generation: usize,
    pub outputs: Vec<[f64; OUT]>,
    pub error: f64,
}

pub struct TrainingsResult<'a, const IN: usize, const OUT: usize> {
    pub input: &'a [f64; IN],
    pub output: [f64; OUT],
    pub generation: usize,
}
