use crate::util::{mean_squarred_error, SetLength};

#[derive(Debug)]
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
    pub fn mean_squarred_error(&self, expected_output: &[f64; OUT]) -> f64 {
        mean_squarred_error(&self.0, expected_output)
    }
}
