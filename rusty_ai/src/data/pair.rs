use std::fmt::Display;

/// an input-expected output-pair.
/// Is used to contain an expected output for training or to contain the output
/// calculated during propagation.
#[derive(Debug, Clone, derive_more::From, derive_more::Into)]
pub struct Pair<const IN: usize, const OUT: usize> {
    pub input: [f64; IN],
    pub output: [f64; OUT],
}

impl<'a, const IN: usize, const OUT: usize> Into<(&'a [f64; IN], &'a [f64; OUT])>
    for &'a Pair<IN, OUT>
{
    fn into(self) -> (&'a [f64; IN], &'a [f64; OUT]) {
        (&self.input, &self.output)
    }
}

impl<const IN: usize, const OUT: usize> Pair<IN, OUT> {
    #[inline]
    pub fn with(
        input: [f64; IN],
        gen_output: impl FnOnce([f64; IN]) -> [f64; OUT],
    ) -> Pair<IN, OUT> {
        Self::from((input, gen_output(input)))
    }
}

impl<const IN: usize, const OUT: usize> Display for Pair<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input: {:?}, expected_output: {:?}", self.input, self.output)
    }
}

// Simple (IN == OUT == 1)

impl From<(f64, f64)> for Pair<1, 1> {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Pair::from(([value.0], [value.1]))
    }
}

impl Pair<1, 1> {
    #[inline]
    pub fn simple_tuple(&self) -> (f64, f64) {
        (self.input[0], self.output[0])
    }
}

impl Into<(f64, f64)> for Pair<1, 1> {
    fn into(self) -> (f64, f64) {
        self.simple_tuple()
    }
}
