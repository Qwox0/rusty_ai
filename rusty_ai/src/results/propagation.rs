pub trait PropResult<const OUT: usize> {
    fn get_nn_output(&self) -> [f64; OUT];
}

#[derive(Debug, derive_more::From, derive_more::Into)]
pub struct PropagationResult<const OUT: usize>(pub [f64; OUT]);

impl<const OUT: usize> From<Vec<f64>> for PropagationResult<OUT> {
    /// # Panics
    /// Panics if the length of `value` is not equal to `OUT`
    fn from(value: Vec<f64>) -> Self {
        assert_eq!(value.len(), OUT);
        let arr: [f64; OUT] = value.try_into().unwrap();
        PropagationResult(arr)
    }
}

impl<const OUT: usize> PropResult<OUT> for PropagationResult<OUT> {
    fn get_nn_output(&self) -> [f64; OUT] {
        self.0
    }
}

/// contains the input and output of every layer
/// caching this data is useful for backpropagation
#[derive(Debug, Clone)]
pub struct VerbosePropagation<const OUT: usize>(Vec<Vec<f64>>);

impl<const OUT: usize> VerbosePropagation<OUT> {
    /// # Panics
    /// Panics if the length of the the last output is not equal to `OUT`.
    pub fn new(vec: Vec<Vec<f64>>) -> Self {
        assert_eq!(vec.last().map(Vec::len), Some(OUT));
        Self(vec)
    }

    pub fn iter_layers<'a>(
        &'a self,
    ) -> impl DoubleEndedIterator<Item = LayerPropagation<'a>> + ExactSizeIterator {
        self.0.array_windows().map(|[input, output]| LayerPropagation { input, output })
    }
}

impl<const OUT: usize> PropResult<OUT> for VerbosePropagation<OUT> {
    fn get_nn_output(&self) -> [f64; OUT] {
        self.0.last().unwrap().as_slice().try_into().unwrap()
    }
}

pub struct LayerPropagation<'a> {
    pub input: &'a Vec<f64>,
    pub output: &'a Vec<f64>,
}
