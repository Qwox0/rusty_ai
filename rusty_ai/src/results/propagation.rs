use crate::util::{constructor, SetLength};

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

pub type LayerOutput = Vec<f64>;

/// contains the output and output derivatives of every layer
/// caching this data is useful for backpropagation
#[derive(Debug, Clone)]
pub struct VerbosePropagation {
    pub outputs: Vec<LayerOutput>,
    pub derivatives: Vec<LayerOutput>,
}

impl VerbosePropagation {
    constructor! { pub new -> outputs: Vec<Vec<f64>>, derivatives: Vec<Vec<f64>> }
}
