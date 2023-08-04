use crate::util::{constructor, SetLength};

#[derive(Debug, derive_more::From, derive_more::Into)]
pub struct PropagationResult<const OUT: usize>(pub [f64; OUT]);

impl<const OUT: usize> From<Vec<f64>> for PropagationResult<OUT> {
    fn from(value: Vec<f64>) -> Self {
        PropagationResult(value.to_arr(f64::default()))
    }
}

/// contains the output and output derivatives of every layer
/// caching this data is useful for backpropagation
#[derive(Debug, Clone)]
pub struct VerbosePropagation {
    pub outputs: Vec<Vec<f64>>,
}

impl VerbosePropagation {
    constructor! { pub new -> outputs: Vec<Vec<f64>> }
}
