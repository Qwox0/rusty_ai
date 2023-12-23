//! # Loss function module

use crate::{gradient::aliases::OutputGradient, layer::Layer, *};
use anyhow::Context;
use derive_more::Display;
use std::{borrow::Borrow, fmt::Display};

/// A trait for calculating the loss from an output of the neural network and the expected output.
///
/// See `rusty_ai::loss_function::*` for some implementations.
pub trait LossFunction<const OUT: usize>: Display {
    /// The type of the expected output used by the loss function.
    ///
    /// For a non default example, see [`NLLLoss`].
    type ExpectedOutput = [f64; OUT];

    /// calculates the loss from an output of the neural network and the expected output.
    fn propagate(
        &self,
        output: &[f64; OUT],
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> f64;

    /*
    #[inline]
    fn propagate(
        &self,
        output: &impl PropResultT<N>,
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> f64 {
        self.propagate_arr(&output.get_nn_output(), expected_output)
    }
    */

    /// calculates the gradient of the Loss with respect to the output of the neural network.
    fn backpropagate_arr(
        &self,
        output: &[f64; OUT],
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> OutputGradient;

    /// calculates the gradient of the Loss with respect to the output of the neural network.
    #[inline]
    fn backpropagate(
        &self,
        output: &VerbosePropagation<OUT>,
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> OutputGradient {
        self.backpropagate_arr(&output.get_nn_output(), expected_output)
    }

    /// validates that the neural network and `Self` are compatible.
    fn check_layers(_layer: &[Layer]) -> anyhow::Result<()> {
        anyhow::Ok(())
    }
}

/// squared error loss function.
/// implements [`LossFunction`].
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct SquaredError;

impl<const N: usize> LossFunction<N> for SquaredError {
    type ExpectedOutput = [f64; N];

    fn propagate(&self, output: &[f64; N], expected_output: impl Borrow<[f64; N]>) -> f64 {
        differences(output, expected_output.borrow()).map(|err| err * err).sum()
    }

    fn backpropagate_arr(
        &self,
        output: &[f64; N],
        expected_output: impl Borrow<[f64; N]>,
    ) -> OutputGradient {
        differences(output, expected_output.borrow()).map(|x| x * 2.0).collect()
    }
}

/// half squared error loss function.
/// implements [`LossFunction`].
///
/// Same as [`SquaredError`] but loss is multiplied by `0.5`.
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct HalfSquaredError;

impl<const N: usize> LossFunction<N> for HalfSquaredError {
    fn propagate(&self, output: &[f64; N], expected_output: impl Borrow<[f64; N]>) -> f64 {
        SquaredError.propagate(output, expected_output) * 0.5
    }

    fn backpropagate_arr(
        &self,
        output: &[f64; N],
        expected_output: impl Borrow<[f64; N]>,
    ) -> OutputGradient {
        differences(output, expected_output.borrow()).collect()
    }
}

/// mean squared error loss function.
/// implements [`LossFunction`].
///
/// Similar to [`SquaredError`] but calculates mean instead of sum
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct MeanSquaredError;

impl<const N: usize> LossFunction<N> for MeanSquaredError {
    fn propagate(&self, output: &[f64; N], expected_output: impl Borrow<[f64; N]>) -> f64 {
        SquaredError.propagate(output, expected_output) / N as f64
    }

    fn backpropagate_arr(
        &self,
        output: &[f64; N],
        expected_output: impl Borrow<[f64; N]>,
    ) -> OutputGradient {
        differences(output, expected_output.borrow())
            .map(|x| x * 2.0 / output.len() as f64)
            .collect()
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
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct NLLLoss;

impl<const OUT: usize> LossFunction<OUT> for NLLLoss {
    type ExpectedOutput = usize;

    /// # Panics
    ///
    /// Panics if `expected_output` is not a valid variant (is not in the range `0..IN`).
    fn propagate(
        &self,
        output: &[f64; OUT],
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> f64 {
        let expected_output = expected_output.borrow();
        assert!((0..OUT).contains(expected_output));
        -output[*expected_output]
    }

    fn backpropagate_arr(
        &self,
        _output: &[f64; OUT],
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> OutputGradient {
        let expected_output = expected_output.borrow();
        let mut gradient = [0.0; OUT];
        gradient[*expected_output] = -1.0;
        gradient.to_vec()
    }

    fn check_layers(layer: &[Layer]) -> Result<(), anyhow::Error> {
        let a = layer
            .last()
            .context("NLLLoss requires at least one layer")?
            .get_activation_function();

        if *a != ActivationFn::LogSoftmax {
            anyhow::bail!(
                "NLLLoss requires log-probabilities as inputs. (hint: use `LogSoftmax` activation \
                 function in the last layer)"
            )
        }

        Ok(())
    }
}

/// Helper function that returns an [`Iterator`] over the differences of elements in `output` and
/// `expected_output`.
#[inline]
fn differences<'a, const N: usize>(
    output: &'a [f64; N],
    expected_output: &'a [f64; N],
) -> impl Iterator<Item = f64> + 'a {
    output.iter().zip(expected_output).map(|(out, expected)| out - expected)
}