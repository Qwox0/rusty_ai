use crate::{
    optimizer::Optimizer,
    prelude::{LossFunction, NNTrainer},
    propagator::SimplePropagator,
};

pub trait HasEO {
    type EO;
    fn get_eo(&self) -> &Self::EO;
}

impl<'a, const IN: usize, EO> HasEO for (&[f64; IN], &EO) {
    type EO = EO;

    fn get_eo(&self) -> &Self::EO {
        self.1
    }
}

pub trait Propagator<const OUT: usize> {
    /// consumes `self` and creates an [`Iterator`] over the network outputs and errors.
    ///
    /// The returned [`Iterator`] *must* be consumed! Otherwise no computations will be performed.
    #[must_use]
    fn outputs_errors(self) -> impl Iterator<Item = ([f64; OUT], f64)>;

    /// consumes `self` and creates an [`Iterator`] over the network outputs.
    ///
    /// The returned [`Iterator`] *must* be consumed! Otherwise no computations will be performed.
    #[must_use]
    #[inline]
    fn outputs(self) -> impl Iterator<Item = [f64; OUT]>
    where Self: Sized {
        self.outputs_errors().map(|(o, _)| o)
    }

    /// consumes `self` and creates an [`Iterator`] over the propagation errors.
    ///
    /// The returned [`Iterator`] *must* be consumed! Otherwise no computations will be performed.
    #[must_use]
    #[inline]
    fn errors(self) -> impl Iterator<Item = f64>
    where Self: Sized {
        self.outputs_errors().map(|(_, e)| e)
    }
}

#[must_use = "`Prop` is lazy and does nothing unless consumed"]
pub struct Prop<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a NNTrainer<IN, OUT, L, O>,
    pairs: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> Prop<'a, IN, OUT, L, O, I> {
    pub(crate) fn new<P>(nn: &'a NNTrainer<IN, OUT, L, O>, pairs: P) -> Self
    where P: IntoIterator<IntoIter = I> {
        Prop { nn, pairs: pairs.into_iter() }
    }
}

impl<'a, const IN: usize, const OUT: usize, L, O, I> Propagator<OUT> for Prop<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT>,
    O: Optimizer,
    I: Iterator<Item = (&'a [f64; IN], &'a L::ExpectedOutput)> + 'a,
{
    fn outputs(self) -> impl Iterator<Item = [f64; OUT]> + 'a {
        self.pairs.map(|(i, _)| self.nn.propagate_arr(i))
    }

    fn outputs_errors(self) -> impl Iterator<Item = ([f64; OUT], f64)> + 'a {
        self.pairs.map(|(input, expected_output)| {
            let out = self.nn.propagate_arr(input);
            let loss = self.nn.calculate_loss_(&out, expected_output);
            (out, loss)
        })
    }
}

#[must_use = "`Backprop` is lazy and does nothing unless consumed"]
pub struct Backprop<'a, const IN: usize, const OUT: usize, L, O, I> {
    nn: &'a mut NNTrainer<IN, OUT, L, O>,
    pairs: I,
}

impl<'a, const IN: usize, const OUT: usize, L, O, EO, I> Backprop<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: Iterator<Item = (&'a [f64; IN], &'a EO)> + 'a,
    EO: 'a,
{
    pub(crate) fn new<P>(nn: &'a mut NNTrainer<IN, OUT, L, O>, pairs: P) -> Self
    where P: IntoIterator<IntoIter = I> {
        Backprop { nn, pairs: pairs.into_iter() }
    }
}
impl<'a, const IN: usize, const OUT: usize, L, O, EO, I> Propagator<OUT>
    for Backprop<'a, IN, OUT, L, O, I>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
    I: Iterator<Item = (&'a [f64; IN], &'a EO)> + 'a,
    EO: 'a,
{
    fn outputs(self) -> impl Iterator<Item = [f64; OUT]> + 'a {
        self.pairs.map(|(input, expected_output)| {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagation(&out, expected_output);
            out.get_nn_output()
        })
    }

    fn outputs_errors(self) -> impl Iterator<Item = ([f64; OUT], f64)> + 'a {
        self.pairs.map(|(input, expected_output)| {
            let out = self.nn.verbose_propagate(input);
            self.nn.backpropagation(&out, expected_output);
            let out = out.get_nn_output();
            let err = self.nn.calculate_loss_(&out, expected_output);
            (out, err)
        })
    }
}

// ======================

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
    ///
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

    pub fn get_nn_output(&self) -> [f64; OUT] {
        self.0.last().unwrap().as_slice().try_into().unwrap()
    }
}

pub struct LayerPropagation<'a> {
    pub input: &'a Vec<f64>,
    pub output: &'a Vec<f64>,
}