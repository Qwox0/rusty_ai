use crate::prelude::*;
use std::{
    f64,
    ops::{Add, Neg},
};

#[derive(Debug, Clone, Copy, Default)]
pub enum ActivationFn {
    /// Identity(x) = x
    /// Identity'(x) = 1
    #[default]
    Identity,

    /// values: (ReLU'(0))
    /// ReLU(x) = max(0, x)
    /// ReLU'(0) := self.0
    ReLU(f64),

    /// values: (leak_rate, LeakyReLU'(0))
    /// LeakyReLU(x) = max(self.0 * x, x)
    /// LeakyReLU'(0) := self.1
    LeakyReLU(f64, f64),

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

#[inline]
fn leaky_relu(x: f64, leak_rate: f64) -> f64 {
    if x.is_sign_positive() { x } else { leak_rate * x }
}

fn softmax(mut vec: Vec<f64>) -> Vec<f64> {
    let mut sum = 0.0;
    for x in vec.iter_mut() {
        *x = x.exp();
        sum += *x;
    }
    vec.into_iter().map(|x| x / sum).collect()
}

fn logsoftmax(vec: Vec<f64>) -> Vec<f64> {
    let ln_sum = vec.iter().copied().map(f64::exp).sum::<f64>().ln();
    vec.into_iter().map(|x| x - ln_sum).collect()
}

impl ActivationFn {
    #[inline]
    pub const fn default_relu() -> ActivationFn {
        ActivationFn::ReLU(1.0)
    }

    #[inline]
    pub const fn default_leaky_relu() -> ActivationFn {
        ActivationFn::LeakyReLU(0.01, 1.0)
    }

    /// Calculates the activation function for a single value [`f64`].
    ///
    /// # Panics
    ///
    /// Panics if the activation function variant requires the entire input Vector.
    pub fn calculate_single(&self, input: f64) -> f64 {
        use ActivationFn::*;
        match self {
            Identity => input,
            ReLU(_) => leaky_relu(input, -0.0),
            LeakyReLU(leak_rate, _) => leaky_relu(input, *leak_rate),
            Sigmoid => input.neg().exp().add(1.0).recip(),

            Softmax => panic!("Softmax needs the entire input Vector"),
            LogSoftmax => panic!("LogSoftmax needs the entire input Vector"),
        }
    }

    /// Calculates the activation function for a single value [`f64`].
    ///
    /// # Panics
    ///
    /// Panics if the activation function variant requires the entire input Vector.
    pub fn derivative(&self, input: f64) -> f64 {
        use ActivationFn::*;
        #[allow(illegal_floating_point_literal_pattern)] // for pattern: 0.0 => ...
        match self {
            Identity => 1.0,
            ReLU(d0) => match input {
                0.0 => *d0,
                x => x.is_sign_positive() as u64 as f64,
            },
            LeakyReLU(leak_rate, d0) => match input {
                0.0 => *d0,
                x if x.is_sign_positive() => 1.0,
                _ => *leak_rate,
            },
            Sigmoid => {
                let exp = input.exp();
                let exp_plus_1 = exp + 1.0;
                exp / (exp_plus_1 * exp_plus_1) // Sigmoid'(x) = e^x/(1+e^x)^2
            },

            Softmax => panic!("Softmax needs the entire input Vector"),
            LogSoftmax => panic!("LogSoftmax needs the entire input Vector"),
        }
    }

    /// Calculates the derivative of the activation function for a single value [`f64`] from the
    /// non-derivative output of the activation function.
    /// # Panics
    /// Panics if the activation function variant only supports entire Vector calculations.
    pub fn derivative_from_output(&self, output: f64) -> f64 {
        use ActivationFn::*;
        #[allow(illegal_floating_point_literal_pattern)] // for pattern: 0.0 => ...
        match self {
            Identity => 1.0,
            ReLU(_) => {
                if output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            LeakyReLU(leak_rate, d0) => match output {
                0.0 => *d0,
                x if x.is_sign_positive() => 1.0,
                _ => *leak_rate,
            },
            Sigmoid => output * (1.0 - output),

            Softmax => panic!("Softmax needs entire Vector"),
            LogSoftmax => panic!("LogSoftmax needs entire Vector"),
        }
    }

    /// Calculates the Vector of neuron activation from `input` which should contain weighted sums.
    ///
    /// # Propagation
    ///
    /// X: Vector of weighted sums
    /// `x_i`: weighted sum of neuron `i`
    /// Y: Vector of neuron acrivations.
    /// `y_i`: activation of neuron `i`.
    /// self: activation function
    ///
    /// General case: `self(X) = Y`
    /// Usual   case: `self(x_i) = y_i` (example: ReLU)
    /// Special case (Softmax): `y_i = e^x_i/(sum of e^x for x in X)`
    pub fn propagate(&self, input: Vec<f64>) -> Vec<f64> {
        use ActivationFn::*;
        match self {
            Identity => input,
            ReLU(_) | LeakyReLU(..) | Sigmoid => input.into_iter().map(self).collect(),
            Softmax => softmax(input),
            LogSoftmax => logsoftmax(input),
        }
    }

    ///
    /// # Propagation
    ///
    /// see `propagate`.
    ///
    /// # Backpropagation
    ///
    /// `L`: total loss of the propagation step.
    ///
    /// General case: `∇_X(L)` = `(J_Y)^T` * `∇_Y(L)`
    /// => `dL/dx_i` = sum `dL/dy` * `dy/dx_i` for y in Y
    /// Usual   case: `dL/dx_i` = `dL/dy_i` * `dy_i/dx_i`
    pub fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
        use ActivationFn::*;
        match self {
            Identity => output_gradient, // `dy_i/dx_i` == 1.0
            ReLU(_) | LeakyReLU(..) | Sigmoid => output_gradient
                .into_iter()
                .zip(self_output)
                .map(|(dl_dy, y)| dl_dy * self.derivative_from_output(*y))
                .collect(),
            Softmax => {
                // dL/dx_i = y_i * (dL/dy_i - sum dL/dy_k * y_k for k in 1..=n)
                let s: f64 =
                    output_gradient.iter().zip(self_output).map(|(&dl_dy, &y)| dl_dy * y).sum();
                // dL/dx_i = y_i * (dL/dy_i - s)
                output_gradient
                    .into_iter()
                    .map(|out_grad| out_grad - s)
                    .zip(self_output)
                    .map(|(out_grad, out)| out * out_grad)
                    .collect()
            },
            LogSoftmax => {
                // dL/dx_i = dL/dy_i - sum dL/dy_k * exp(y_k) for k in 1..=n
                let s: f64 = self_output
                    .iter()
                    .copied()
                    .map(f64::exp)
                    .zip(&output_gradient)
                    .map(|(exp_y, dl_dy)| dl_dy * exp_y)
                    .sum();
                // dL/dx_i = dL/dy_i - s
                output_gradient.into_iter().map(|dl_dy| dl_dy - s).collect()
            },
        }
    }

    #[inline]
    fn call_single(&self, input: (f64,)) -> f64 {
        self.calculate_single(input.0)
    }

    #[inline]
    fn call_vec(&self, input: (Vec<f64>,)) -> Vec<f64> {
        self.propagate(input.0)
    }
}

impl_fn_traits! { ActivationFn : call_single => Fn<(f64,)> -> f64 }
impl_fn_traits! { ActivationFn : call_vec => Fn<(Vec<f64>,)> -> Vec<f64> }

impl std::fmt::Display for ActivationFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ActivationFn::*;
        match self {
            Identity => write!(f, "Identity"),
            ReLU(d0) => write!(f, "ReLU (ReLU'(0)={})", d0),
            LeakyReLU(a, d0) => write!(f, "Leaky ReLU (a={}; f'(0)={})", a, d0),
            Sigmoid => write!(f, "Sigmoid"),
            Softmax => write!(f, "Softmax"),
            LogSoftmax => write!(f, "LogSoftmax"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_wiki() {
        let data = vec![1, 2, 3, 4, 1, 2, 3].into_iter().map(f64::from).collect();
        let data = softmax(data);
        println!("{:?}", data);
        let rounded: Vec<_> = data.iter().map(|x| (1000.0 * x).round() / 1000.0).collect();
        println!("{:?}", rounded);
        assert_eq!(rounded, &[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]);
    }

    #[test]
    fn test_relu() {
        const D0: f64 = 0.5;
        use std::f64::consts::PI;
        let relu = ActivationFn::ReLU(D0);

        assert_eq!(relu.calculate_single(-10.0), 0.0);
        assert_eq!(relu.calculate_single(0.0), 0.0);
        assert_eq!(relu.calculate_single(1.0), 1.0);
        assert_eq!(relu.calculate_single(PI), PI);
        assert_eq!(relu.calculate_single(10.0), 10.0);

        assert_eq!(relu.derivative(-10.0), 0.0);
        assert_eq!(relu.derivative(0.0), D0);
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(PI), 1.0);
        assert_eq!(relu.derivative(10.0), 1.0);
    }
}

#[allow(unused_imports)]
mod benches {
    extern crate test;

    use super::*;
    use test::{black_box, Bencher};

    macro_rules! make_bench {
        ($( $bench:ident : $fn:ident )*) => {
            $(
                #[bench]
                fn $bench(b: &mut Bencher) {
                    let mut vec = vec![0.0; 10000];

                    use rand::Rng;

                    rand::thread_rng().fill(vec.as_mut_slice());

                    b.iter(|| {
                        black_box($fn(black_box(vec.clone())))
                    });
                }
             )*

        };
    }

    make_bench! {
        bench_softmax: softmax
        bench_logsoftmax: logsoftmax
    }

    #[bench]
    fn muladd(bench: &mut Bencher) {
        let mut a = vec![0.0; 100000];
        let mut b = vec![0.0; 100000];
        let mut c = vec![0.0; 100000];

        use rand::Rng;

        rand::thread_rng().fill(a.as_mut_slice());
        rand::thread_rng().fill(b.as_mut_slice());
        rand::thread_rng().fill(c.as_mut_slice());

        bench.iter(|| {
            for i in 0..a.len() {
                black_box(f64::mul_add(black_box(a[i]), black_box(b[i]), black_box(c[i])));
            }
        });
    }

    #[bench]
    fn manual_muladd(bench: &mut Bencher) {
        let mut a = vec![0.0; 100000];
        let mut b = vec![0.0; 100000];
        let mut c = vec![0.0; 100000];

        use rand::Rng;

        rand::thread_rng().fill(a.as_mut_slice());
        rand::thread_rng().fill(b.as_mut_slice());
        rand::thread_rng().fill(c.as_mut_slice());

        fn mul_add(a: f64, b: f64, c: f64) -> f64 {
            a * b + c
        }

        bench.iter(|| {
            for i in 0..a.len() {
                black_box(mul_add(black_box(a[i]), black_box(b[i]), black_box(c[i])));
            }
        });
    }
}
