use const_tensor::{tensor, Element, Shape, Tensor};

/// A pair containing an input and an expected output.
#[derive(Debug, Clone, Default)]
pub struct Pair<X: Element, IN: Shape, EO> {
    input: Tensor<X, IN>,
    expected_output: EO,
}

impl<X: Element, IN: Shape, EO> Pair<X, IN, EO> {
    /// Creates a new input, expected output [`Pair`].
    pub fn new(input: Tensor<X, IN>, expected_output: EO) -> Self {
        Self { input: input.into(), expected_output: expected_output.into() }
    }

    /// Returns the input.
    pub fn get_input(&self) -> &tensor<X, IN> {
        &self.input
    }

    /// Returns the expected output.
    pub fn get_expected_output(&self) -> &EO {
        &self.expected_output
    }

    /// Converts the [`Pair`] into a tuple.
    pub fn as_tuple(&self) -> (&tensor<X, IN>, &EO) {
        (self.get_input(), self.get_expected_output())
    }

    /// Converts the [`Pair`] into a tuple.
    pub fn into_tuple(self) -> (Tensor<X, IN>, EO) {
        (self.input, self.expected_output)
    }
}

impl<X: Element, IN: Shape, EO> From<(Tensor<X, IN>, EO)> for Pair<X, IN, EO> {
    fn from(value: (Tensor<X, IN>, EO)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<X: Element, IN: Shape, EO> From<Pair<X, IN, EO>> for (Tensor<X, IN>, EO) {
    fn from(pair: Pair<X, IN, EO>) -> Self {
        pair.into_tuple()
    }
}

impl<'a, X: Element, IN: Shape, EO> From<&'a Pair<X, IN, EO>> for (&'a tensor<X, IN>, &'a EO) {
    fn from(pair: &'a Pair<X, IN, EO>) -> Self {
        pair.as_tuple()
    }
}
