use std::fmt::Display;

/// an Input-Output-pair.
/// Is used to contain an expected output for training or to contain the output calculated during
/// propagation.
#[derive(Debug, Clone)]
pub struct DataPair<const IN: usize, const OUT: usize> {
    pub input: [f64; IN],
    pub output: [f64; OUT],
}

impl<const IN: usize, const OUT: usize> From<([f64; IN], [f64; OUT])> for DataPair<IN, OUT> {
    fn from(value: ([f64; IN], [f64; OUT])) -> Self {
        let (input, output) = value;
        DataPair { input, output }
    }
}

impl<const IN: usize, const OUT: usize> Into<([f64; IN], [f64; OUT])> for DataPair<IN, OUT> {
    fn into(self) -> ([f64; IN], [f64; OUT]) {
        (self.input, self.output)
    }
}

impl<'a, const IN: usize, const OUT: usize> Into<(&'a [f64; IN], &'a [f64; OUT])>
    for &'a DataPair<IN, OUT>
{
    fn into(self) -> (&'a [f64; IN], &'a [f64; OUT]) {
        (&self.input, &self.output)
    }
}

impl<const IN: usize, const OUT: usize> DataPair<IN, OUT> {
    pub fn random(get_out: impl FnOnce([f64; IN]) -> [f64; OUT]) -> DataPair<IN, OUT> {
        let mut x = [0.0; IN];
        x.iter_mut().for_each(|x| *x = rand::random());
        (x, get_out(x)).into()
    }
}

impl<const IN: usize, const OUT: usize> Display for DataPair<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input: {:?}, expected_output: {:?}",
            self.input, self.output
        )
    }
}

// Simple (IN == OUT == 1)

impl From<(f64, f64)> for DataPair<1, 1> {
    fn from(value: (f64, f64)) -> Self {
        DataPair::from(([value.0], [value.1]))
    }
}

impl DataPair<1, 1> {
    pub fn random_simple(get_out: impl FnOnce(f64) -> f64) -> DataPair<1, 1> {
        let x = rand::random();
        (x, get_out(x)).into()
    }

    pub fn simple_tuple(&self) -> (f64, f64) {
        (self.input[0], self.output[0])
    }
}

impl Into<(f64, f64)> for DataPair<1, 1> {
    fn into(self) -> (f64, f64) {
        (self.input[0], self.output[0])
    }
}
