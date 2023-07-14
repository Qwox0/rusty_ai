use rand::{distributions::uniform::SampleRange, Rng};
use std::fmt::Display;

/// an input-expected output-pair.
/// Is used to contain an expected output for training or to contain the output
/// calculated during propagation.
#[derive(Debug, Clone)]
pub struct Pair<const IN: usize, const OUT: usize> {
    pub input: [f64; IN],
    pub output: [f64; OUT],
}

impl<const IN: usize, const OUT: usize> From<([f64; IN], [f64; OUT])> for Pair<IN, OUT> {
    fn from(value: ([f64; IN], [f64; OUT])) -> Self {
        let (input, output) = value;
        Pair { input, output }
    }
}

impl<const IN: usize, const OUT: usize> Into<([f64; IN], [f64; OUT])> for Pair<IN, OUT> {
    fn into(self) -> ([f64; IN], [f64; OUT]) { (self.input, self.output) }
}

impl<'a, const IN: usize, const OUT: usize> Into<(&'a [f64; IN], &'a [f64; OUT])>
    for &'a Pair<IN, OUT>
{
    fn into(self) -> (&'a [f64; IN], &'a [f64; OUT]) { (&self.input, &self.output) }
}

impl<const IN: usize, const OUT: usize> Pair<IN, OUT> {
    pub fn gen(
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
    fn from(value: (f64, f64)) -> Self { Pair::from(([value.0], [value.1])) }
}

impl Pair<1, 1> {
    pub fn random_simple(
        range: impl SampleRange<f64>,
        get_out: impl FnOnce(f64) -> f64,
    ) -> Pair<1, 1> {
        let x = rand::thread_rng().gen_range(range);
        (x, get_out(x)).into()
    }

    pub fn simple_tuple(&self) -> (f64, f64) { (self.input[0], self.output[0]) }
}

impl Into<(f64, f64)> for Pair<1, 1> {
    fn into(self) -> (f64, f64) { (self.input[0], self.output[0]) }
}
