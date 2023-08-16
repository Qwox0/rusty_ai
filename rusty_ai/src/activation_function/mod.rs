mod relu;
mod sigmoid;
mod softmax;

use crate::prelude::*;
use derive_more::Display;

/// see the specific struct for documentation.
#[derive(Debug, Clone, Copy, Display, Default)]
pub enum ActivationFn {
    /// Identity(x) = x
    /// Identity'(x) = 1
    #[default]
    Identity,

    /// ReLU(x) = max(0, x)
    ///
    /// # Derivative
    ///
    /// ReLU'(0) := 0
    /// see [Numerical influence of ReLU'(0) on backpropagation](https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf)
    ReLU,

    /// LeakyReLU(x) = max(leak_rate * x, x)
    ///
    /// # Derivative
    ///
    /// LeakyReLU'(0) := 0
    /// see [Numerical influence of ReLU'(0) on backpropagation](https://hal.science/hal-03265059/file/Impact_of_ReLU_prime.pdf)
    LeakyReLU { leak_rate: f64 },

    /// Sigmoid(x) = 1/(1 + exp(-x)) = exp(x)/(exp(x) + 1)
    /// Sigmoid'(x) = e^(-x)/(1+e^(-x))^2 = e^x/(1+e^x)^2
    Sigmoid,

    /// f64^n -> f64^n
    /// where `X` ↦ `Y`
    /// where `x_i` ↦ `y_i` = `e^x_i`/(sum `e^x` for x in X)
    ///
    /// # Jacobian of Softmax(X):
    ///
    /// ┌` y_1(1-y_1)` `-y_1*y_2   ` `…` `-y_1*y_n   `┐
    /// │`-y_1*y_2   ` ` y_2(1-y_2)` `…` `-y_2*y_n   `│
    /// │`    …` `          …` `      …` `    …`      │
    /// └`-y_1*y_n   ` `-y_2*y_n   ` `…` ` y_n(1-y_n)`┘
    ///
    /// # Backpropagation
    ///
    /// `∇_X(L)` = `(J_Y)^T` * `∇_Y(L)` => `dL/dx_i` = sum `dL/dy` * `dy/dx_i` for y in Y
    ///
    /// _dL/dx_i_
    /// = _dL/dy_1_ * _dy_1/x_i_ `+` _dL/dy_2_ * _dy_2/x_i_ `+` … `+` _dL/dy_n_ * _dy_n/x_i_
    /// = `sum` _dL/dy_k_ * _dy_k/x_i_ for k in `1..=n`
    /// = _dL/dy_i_ * `dy_i/x_i` + sum _dL/dy_k_ * `dy_k/x_i` for k in 1..=n and k != i
    /// = _dL/dy_i_ * `y_i`_(1-y_i)_ `+` sum _dL/dy_k_ * `-y_i`*_y_k_ for k in 1..=n and k != i
    /// = _y_i_ * (_dL/dy_i_ * `(1-y_i)` - sum _dL/dy_k_ * _y_k_ for k in 1..=n and `k != i`)
    /// = _y_i_ * (_dL/dy_i_ - sum _dL/dy_k_ * _y_k_ for k in 1..=n)
    Softmax,

    /// f64^n -> f64^n
    /// where `X` ↦ `Y`
    /// where `x_i` ↦ `y_i` = `ln(e^x_i/(sum e^x for x in X))` = `x_i - ln(sum e^x for x in X)`
    ///
    /// `LogSoftmax(X)` = `ln(Softmax(X))`
    ///
    /// # Jacobian of LogSoftmax(X):
    ///
    /// ┌`1-exp(y_1)`  ` -exp(y_2)`  `…`  ` -exp(y_n)`┐
    /// │` -exp(y_1)`  `1-exp(y_2)`  `…`  ` -exp(y_n)`│
    /// │`     …` `          …` `     …`  `     …`    │
    /// └` -exp(y_1)`  ` -exp(y_2)`  `…`  `1-exp(y_n)`┘
    ///
    /// `w_i` := `e^x_i`/(sum `e^x` for x in X)
    /// `w_i` = `e^y_i` = `Softmax(x_i)`
    ///
    /// ┌`1-w_1`  ` -w_2`  `…`  ` -w_n`┐
    /// │` -w_1`  `1-w_2`  `…`  ` -w_n`│
    /// │`  …` `     …` `   …`  `  …`  │
    /// └` -w_1`  ` -w_2`  `…`  `1-w_n`┘
    ///
    /// # Backpropagation
    ///
    /// see [`ActivationFn::Softmax`].
    ///
    /// `dL/dx_i`
    /// = _dL/dy_i_ * `1-exp(y_i)` `+` sum _dL/dy_k_ * `-exp(y_k)` for k in 1..=n and k != i
    /// = _dL/dy_i_ - sum _dL/dy_k_ * `exp(y_k)` for k in 1..=n
    LogSoftmax,
}

impl ActivationFunction for ActivationFn {
    fn propagate(&self, input: Vec<f64>) -> Vec<f64> {
        use ActivationFn::*;
        match *self {
            Identity => ActivationFunction::propagate(&identity::Identity, input),
            ReLU => ActivationFunction::propagate(&relu::ReLU, input),
            LeakyReLU { leak_rate } => {
                ActivationFunction::propagate(&relu::LeakyReLU { leak_rate }, input)
            },
            Sigmoid => ActivationFunction::propagate(&sigmoid::Sigmoid, input),
            Softmax => ActivationFunction::propagate(&softmax::Softmax, input),
            LogSoftmax => ActivationFunction::propagate(&softmax::LogSoftmax, input),
        }
    }

    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
        todo!()
    }
}

/// Helper for [`ActivationFn`].
pub trait ActivationFunction {
    /// Calculates the Vector of neuron activation from `input` which should contain weighted sums.
    ///
    /// # Propagation
    ///
    /// `X`: Vector of weighted sums
    /// `x_i`: weighted sum of neuron `i`
    /// `Y`: Vector of neuron acrivations.
    /// `y_i`: activation of neuron `i`.
    /// self: activation function
    ///
    /// General case: `self(X) = Y`
    /// Usual   case: `self(x_i) = y_i` (example: ReLU)
    /// Special case: `y_i = e^x_i/(sum of e^x for x in X)` (example: Softmax)
    fn propagate(&self, input: Vec<f64>) -> Vec<f64>;

    /// # Propagation
    ///
    /// see [`propagate`](ActivationFunction::propagate).
    ///
    /// # Backpropagation
    ///
    /// `X`: Vector of weighted sums (`self_input`)
    /// `x_i`: weighted sum of neuron `i`
    /// `Y`: Vector of neuron acrivations (`self_output`)
    /// `y_i`: activation of neuron `i`.
    /// `L`: total loss of the propagation step.
    /// `∇_Y(L)`: `output_gradient`
    /// `∇_X(L)`: [`WeightedSumGradient`]
    ///
    /// General case: `∇_X(L)` = `(J_Y)^T` * `∇_Y(L)`
    /// => `dL/dx_i` = sum `dL/dy` * `dy/dx_i` for y in Y
    /// Simple  case: `dL/dx_i` = `dL/dy_i` * `dy_i/dx_i`
    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient;
}

/// Similar to [`ActivationFunction`] but function can be applied to a single [`f64`]. In case of
/// an input Vector each value is independently mapped to the output Vector.
///
/// Each type implementing this trait automatically implements [`ActivationFunction`] with the
/// behavior described above.
trait SimpleActivationFunction: ActivationFunction {
    fn propagate(&self, input: f64) -> f64;

    /// Calculates the derivative of the activation function for a single value [`f64`] from the
    /// non-derivative output of the activation function.
    /// # Panics
    /// Panics if the activation function variant only supports entire Vector calculations.
    fn derivative_from_output(&self, self_output: f64) -> f64;

    /// # Backpropagation
    ///
    /// see [`ActivationFunction::backpropagate`]
    /// -> Simple case: `dL/dx_i` = `dL/dy_i` * `dy_i/dx_i`
    ///
    /// `x_i`: weighted sum of neuron `i` (`self_input`)
    /// `y_i`: activation of neuron `i`. (`self_output`)
    /// `dL/dy_i`: `output_gradient`
    #[inline]
    fn backpropagate(&self, output_derivative: f64, self_output: f64) -> f64 {
        output_derivative * self.derivative_from_output(self_output)
    }
}

impl<T: SimpleActivationFunction> ActivationFunction for T {
    fn propagate(&self, input: Vec<f64>) -> Vec<f64> {
        let prop = |x| SimpleActivationFunction::propagate(self, x);
        input.into_iter().map(prop).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
        output_gradient
            .into_iter()
            .zip(self_output)
            .map(|(dl_dy, y)| SimpleActivationFunction::backpropagate(self, dl_dy, *y))
            .collect()
    }
}

mod identity {
    use super::*;

    #[derive(Debug, Clone, Copy, Display)]
    pub(super) struct Identity;

    impl SimpleActivationFunction for Identity {
        fn propagate(&self, input: f64) -> f64 {
            input
        }

        fn derivative_from_output(&self, _self_output: f64) -> f64 {
            1.0
        }
    }
}
