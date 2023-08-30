use std::fmt::Display;

/// an input-expected output-pair.
/// Is used to contain an expected output for training or to contain the output
/// calculated during propagation.
#[derive(Debug, Clone, derive_more::From, derive_more::Into)]
pub struct Pair<const IN: usize, EO> {
    pub input: [f64; IN],
    pub expected_output: EO,
}

impl<'a, const IN: usize, EO> Into<(&'a [f64; IN], &'a EO)> for &'a Pair<IN, EO> {
    fn into(self) -> (&'a [f64; IN], &'a EO) {
        (&self.input, &self.expected_output)
    }
}

impl<'a, const IN: usize, EO> Into<&'a [f64; IN]> for &'a Pair<IN, EO> {
    fn into(self) -> &'a [f64; IN] {
        &self.input
    }
}

impl<const IN: usize, EO> Pair<IN, EO> {
    pub fn new(input: [f64; IN], expected_output: EO) -> Pair<IN, EO> {
        Pair { input, expected_output }
    }

    #[inline]
    pub fn with(input: [f64; IN], gen_output: impl FnOnce([f64; IN]) -> EO) -> Pair<IN, EO> {
        Self::from((input, gen_output(input)))
    }
}

impl<const IN: usize, EO: Display> Display for Pair<IN, EO> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input: {:?}, expected_output: {}", self.input, self.expected_output)
    }
}

// Simple (IN == OUT == 1)

impl From<(f64, f64)> for Pair<1, f64> {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Pair::from(([value.0], value.1))
    }
}

impl Pair<1, f64> {
    #[inline]
    pub fn simple_tuple(&self) -> (f64, f64) {
        (self.input[0], self.expected_output)
    }
}

impl Into<(f64, f64)> for Pair<1, f64> {
    fn into(self) -> (f64, f64) {
        self.simple_tuple()
    }
}
