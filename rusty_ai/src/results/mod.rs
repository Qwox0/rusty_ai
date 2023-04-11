mod gradient_layer;
mod propagation;

pub(crate) use gradient_layer::GradientLayer;
pub use propagation::PropagationResult;

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
