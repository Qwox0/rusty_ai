use crate::{
    prelude::Pair,
    traits::{NNInput, Propagator},
};

pub struct InputPropagation<'a, NN, I, const IN: usize>
where I: Iterator<Item = &'a [f64; IN]>
{
    nn: &'a NN,
    inputs: I,
}

impl<'a, NN, I, const IN: usize, const OUT: usize, EO> InputPropagation<'a, NN, I, IN>
where
    NN: Propagator<IN, OUT, EO>,
    I: Iterator<Item = &'a [f64; IN]>,
{
    pub fn new(nn: &'a NN, inputs: impl IntoIterator<IntoIter = I>) -> Self {
        InputPropagation { nn, inputs: inputs.into_iter() }
    }

    pub fn outputs(self) -> impl Iterator<Item = [f64; OUT]> {
        self.inputs.map(|i| self.nn.propagate_single(i))
    }

    pub fn expected_outputs(
        self,
        expected_outputs: impl IntoIterator<Item = &'a EO>,
    ) -> PairPropagation<'a, NN, impl Iterator<Item = &'a Pair<IN, EO>>, IN, EO> {
        let pairs = self.inputs.zip(expected_outputs).map(Pair::from);
        PairPropagation { nn: self.nn, pairs }
    }
}

/// lazy
pub struct PairPropagation<'a, NN, I, const IN: usize, EO>
where I: Iterator<Item = &'a Pair<IN, EO>>
{
    nn: &'a NN,
    pairs: I,
}

impl<'a, NN, I, const IN: usize, const OUT: usize, EO> PairPropagation<'a, NN, I, IN, EO>
where
    NN: Propagator<IN, OUT, EO>,
    I: Iterator<Item = &'a Pair<IN, EO>>,
{
    pub fn new(nn: &'a NN, pairs: impl IntoIterator<IntoIter = I>) -> Self {
        PairPropagation { nn, pairs: pairs.into_iter() }
    }

    pub fn outputs(self) -> impl Iterator<Item = [f64; OUT]> {
        self.inputs.map(|i| self.nn.propagate_single(i))
    }

    pub fn error(self) -> f64 {
        todo!()
    }

    pub fn output_with_error(self) -> ([f64; OUT], f64) {
        todo!()
    }

    pub fn backpropagate(self) -> [f64; OUT] {
        todo!()
    }
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

/*
impl<const OUT: usize> PropResultT<OUT> for VerbosePropagation<OUT> {
    fn get_nn_output(&self) -> [f64; OUT] {
        self.0.last().unwrap().as_slice().try_into().unwrap()
    }
}
*/

pub struct LayerPropagation<'a> {
    pub input: &'a Vec<f64>,
    pub output: &'a Vec<f64>,
}
