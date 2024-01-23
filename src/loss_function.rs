//! # Loss function module

use const_tensor::{tensor, vector, Element, Float, Len, Num, Shape, Tensor, TensorData, Vector};
use derive_more::Display;

/// A trait for calculating the loss from an output of the neural network and the expected output.
///
/// See `rusty_ai::loss_function::*` for some implementations.
///
/// `vector<X,N>`: output dimension of the [`NeuralNetwork`].
pub trait LossFunction<X: Element, S: Shape>: Send + Sync + 'static {
    /// The type of the expected output used by the loss function.
    ///
    /// For a non default example, see [`NLLLoss`].
    type ExpectedOutput = tensor<X, S>;

    /// calculates the loss from an output of the neural network and the expected output.
    fn propagate(&self, output: &tensor<X, S>, expected_output: &Self::ExpectedOutput) -> X;

    /*
    #[inline]
    fn propagate(
        &self,
        output: &impl PropResultT<N>,
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> X {
        self.propagate_arr(&output.get_nn_output(), expected_output)
    }
    */

    /// calculates the gradient of the loss with respect to the output of the neural network.
    fn backpropagate(
        &self,
        output: &tensor<X, S>,
        expected_output: &Self::ExpectedOutput,
    ) -> Tensor<X, S>;

    /*
    /// validates that the neural network and `Self` are compatible.
    fn check_layers(_layer: &[Layer<X>]) -> anyhow::Result<()> {
        anyhow::Ok(())
    }
    */
}

/// squared error loss function.
/// implements [`LossFunction`].
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct SquaredError;

impl<X: Num, S: Shape> LossFunction<X, S> for SquaredError
where S: Len<{ S::LEN }>
{
    fn propagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> X {
        differences(output, expected_output).map(|err| err * err).sum()
    }

    fn backpropagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> Tensor<X, S> {
        Tensor::from_iter(differences(output, expected_output).map(|x| x * X::lit(2)))
    }
}

/// half squared error loss function.
/// implements [`LossFunction`].
///
/// Same as [`SquaredError`] but loss is multiplied by `0.5`.
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct HalfSquaredError;

impl<X: Float, S: Shape> LossFunction<X, S> for HalfSquaredError
where S: Len<{ S::LEN }>
{
    fn propagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> X {
        SquaredError.propagate(output, expected_output) * X::f_lit(0.5)
    }

    fn backpropagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> Tensor<X, S> {
        Tensor::from_iter(differences(output, expected_output))
    }
}

/// mean squared error loss function.
/// implements [`LossFunction`].
///
/// Similar to [`SquaredError`] but calculates mean instead of sum
#[derive(Debug, Clone, Copy, Default, Display)]
pub struct MeanSquaredError;

impl<X: Num, S: Shape> LossFunction<X, S> for MeanSquaredError
where S: Len<{ S::LEN }>
{
    fn propagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> X {
        SquaredError.propagate(output, expected_output) / S::LEN.cast()
    }

    fn backpropagate(&self, output: &tensor<X, S>, expected_output: &tensor<X, S>) -> Tensor<X, S> {
        Tensor::from_iter(
            differences(output, expected_output).map(|x| x * X::lit(2) / S::LEN.cast()),
        )
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

impl<X: Float, const N: usize> LossFunction<X, [(); N]> for NLLLoss {
    type ExpectedOutput = usize;

    /// # Panics
    ///
    /// Panics if `expected_output` is not a valid variant (is not in the range `0..IN`).
    fn propagate(&self, output: &vector<X, N>, expected_output: &Self::ExpectedOutput) -> X {
        assert!((0..N).contains(expected_output));
        output[*expected_output].val().neg()
    }

    fn backpropagate(
        &self,
        _output: &vector<X, N>,
        expected_output: &Self::ExpectedOutput,
    ) -> Vector<X, N> {
        let mut gradient = Vector::new([X::zero(); N]);
        gradient[*expected_output].set(-X::ONE);
        gradient
    }

    /*
    fn check_layers(layer: &[Layer<X>]) -> Result<(), anyhow::Error> {
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
    */
}

/// Helper function that returns an [`Iterator`] over the differences of elements in `output` and
/// `expected_output`.
#[inline]
fn differences<'a, X: Num, S: Shape>(
    output: &'a tensor<X, S>,
    expected_output: &'a tensor<X, S>,
) -> impl Iterator<Item = X> + 'a
where
    S: Len<{ S::LEN }>,
{
    output
        .iter_elem()
        .zip(expected_output.iter_elem())
        .map(|(out, expected)| *out - *expected)
}
