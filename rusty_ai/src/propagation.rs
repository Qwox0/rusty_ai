use std::slice::ArrayWindows;

/// contains the input and output of every layer
/// caching this data is useful for backpropagation
#[derive(Debug, Clone)]
pub struct VerbosePropagation<const OUT: usize>(Vec<Vec<f64>>);

impl<const OUT: usize> VerbosePropagation<OUT> {
    /// # Panics
    ///
    /// Panics if the length of the the last output is not equal to `OUT`.
    pub fn new(vec: Vec<Vec<f64>>) -> Self {
        assert_eq!(vec.last().map(Vec::len), Some(OUT));
        Self(vec)
    }

    /// Returns an [`Iterator`] over the input and output of every layer.
    pub fn iter_layers<'a>(&'a self) -> ArrayWindows<'a, Vec<f64>, 2> {
        self.0.array_windows()
    }

    /// TODO?: HeapArray output?
    pub fn get_nn_output(&self) -> [f64; OUT] {
        self.0.last().unwrap().as_slice().try_into().unwrap()
    }
}
