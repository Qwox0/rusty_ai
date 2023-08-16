use crate::prelude::*;
use derive_more::Display;
use std::fmt::Display;

/// See `rusty_ai::loss_function::*` for some implementations.
pub trait LossFunction<const N: usize>: Display {
    type ExpectedOutput;

    fn propagate(&self, output: &[f64; N], expected_output: &Self::ExpectedOutput) -> f64;

    fn backpropagate(
        &self,
        output: &[f64; N],
        expected_output: &Self::ExpectedOutput,
    ) -> OutputGradient;
}

/// squared error loss function.
/// implements [`LossFunction`].
#[derive(Debug, Clone, Copy, Display)]
pub struct SquaredError;

impl<const N: usize> LossFunction<N> for SquaredError {
    type ExpectedOutput = [f64; N];

    fn propagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> f64 {
        squared_errors(output, expected_output).sum()
    }

    fn backpropagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> OutputGradient {
        differences(output, expected_output).map(|x| x * 2.0).collect()
    }
}

/// half squared error loss function.
/// implements [`LossFunction`].
///
/// Same as [`SquaredError`] but loss is multiplied by `0.5`.
#[derive(Debug, Clone, Copy, Display)]
pub struct HalfSquaredError;

impl<const N: usize> LossFunction<N> for HalfSquaredError {
    type ExpectedOutput = [f64; N];

    fn propagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> f64 {
        SquaredError.propagate(output, expected_output) * 0.5
    }

    fn backpropagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> OutputGradient {
        differences(output, expected_output).collect()
    }
}

/// mean squared error loss function.
/// implements [`LossFunction`].
///
/// Similar to [`SquaredError`] but calculates mean instead of sum
#[derive(Debug, Clone, Copy, Display)]
pub struct MeanSquaredError;

impl<const N: usize> LossFunction<N> for MeanSquaredError {
    type ExpectedOutput = [f64; N];

    fn propagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> f64 {
        SquaredError.propagate(output, expected_output) / N as f64
    }

    fn backpropagate(&self, output: &[f64; N], expected_output: &[f64; N]) -> OutputGradient {
        differences(output, expected_output).map(|x| x * 2.0 / output.len() as f64).collect()
    }
}

/// The negative log likelihood loss.
/// implements [`LossFunction`].
/// See [`https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html`].
///
/// The network output is expected to contain log-probabilities of each class.
/// Use [`ActivationFn::LogSoftmax`] as the activation function of the last layer.
///
/// # Propagation
///
/// `L` = `-o_e`            // `-w_e * o_e`
///
/// # Backpropagation
///
/// `dL/do_i` = `0` if `i != e`
/// `dL/do_e` = `-1`        // `-w_e`
///
/// # Description
///
/// `o_i`: network output `i`
/// `N`: number of network outputs
/// `e`: expected variant/class (unsigned integer in range `0..N`)
/// `L`: Loss
#[derive(Debug, Clone, Copy, Display)]
pub struct NLLLoss;

impl<const OUT: usize> LossFunction<OUT> for NLLLoss {
    type ExpectedOutput = usize;

    /// # Panics
    ///
    /// Panics if `expected_output` is not a valid variant (is not in the range `0..IN`).
    fn propagate(&self, output: &[f64; OUT], expected_output: &Self::ExpectedOutput) -> f64 {
        assert!((0..OUT).contains(expected_output));
        -output[*expected_output]
    }

    fn backpropagate(
        &self,
        _output: &[f64; OUT],
        expected_output: &Self::ExpectedOutput,
    ) -> OutputGradient {
        let mut gradient = [0.0; OUT];
        gradient[*expected_output] = -1.0;
        gradient.to_vec()
    }
}

// Helper

#[inline]
fn differences<'a, const N: usize>(
    output: &'a [f64; N],
    expected_output: &'a [f64; N],
) -> impl Iterator<Item = f64> + 'a {
    output.iter().zip(expected_output).map(|(out, expected)| out - expected)
}

#[inline]
fn squared_errors<'a, const N: usize>(
    output: &'a [f64; N],
    expected_output: &'a [f64; N],
) -> impl Iterator<Item = f64> + 'a {
    differences(output, expected_output).map(|err| err * err)
}
