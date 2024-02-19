//! # Test Result Module
//!
//! contains the [`TestResult`] struct

use const_tensor::{Element, Shape, Tensor};

/// Result of the test of a nn.
///
/// This contains the calculated output and loss.
#[derive(Debug, Clone, Default)]
pub struct TestResult<X: Element, OUT: Shape> {
    output: Tensor<X, OUT>,
    loss: X,
}

impl<X: Element, OUT: Shape> TestResult<X, OUT> {
    /// Creates a new [`TestResult`].
    pub fn new(output: Tensor<X, OUT>, loss: X) -> Self {
        Self { output, loss }
    }

    /// Returns the output of the Test
    pub fn get_output(&self) -> &Tensor<X, OUT> {
        &self.output
    }

    /// Returns the loss of the Test
    pub fn get_loss(&self) -> X {
        self.loss
    }

    /// Converts the [`TestResult`] into a tuple of output and loss.
    pub fn into_tuple(self) -> (Tensor<X, OUT>, X) {
        (self.output, self.loss)
    }
}

impl<X: Element, OUT: Shape> From<TestResult<X, OUT>> for (Tensor<X, OUT>, X) {
    fn from(res: TestResult<X, OUT>) -> Self {
        res.into_tuple()
    }
}
