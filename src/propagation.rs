use matrix::Element;
use std::slice::ArrayWindows;

/// contains the input and output of every layer
/// caching this data is useful for backpropagation
#[derive(Debug, Clone)]
pub struct VerbosePropagation<X, const OUT: usize>(Vec<Vec<X>>);

impl<X, const OUT: usize> VerbosePropagation<X, OUT> {
    /// # Panics
    ///
    /// Panics if the length of the the last output is not equal to `OUT`.
    pub fn new(vec: Vec<Vec<X>>) -> Self {
        assert_eq!(vec.last().map(Vec::len), Some(OUT));
        Self(vec)
    }

    /// Returns an [`Iterator`] over the input and output of every layer.
    pub fn iter_layers<'a>(&'a self) -> ArrayWindows<'a, Vec<X>, 2> {
        self.0.array_windows()
    }
}

impl<X: Element, const OUT: usize> VerbosePropagation<X, OUT> {
    /// TODO?: HeapArray output?
    pub fn get_nn_output(&self) -> [X; OUT] {
        self.0.last().unwrap().as_slice().try_into().unwrap()
    }
}
