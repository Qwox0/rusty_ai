use super::SimpleActivationFunction;
use derive_more::Display;

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct ReLU;

impl SimpleActivationFunction for ReLU {
    #[inline]
    fn propagate(&self, input: f64) -> f64 {
        leaky_relu(input, 0.0)
    }

    #[inline]
    fn derivative_from_output(&self, self_output: f64) -> f64 {
        if self_output.is_sign_positive() { 1.0 } else { 0.0 }
    }
}

#[derive(Debug, Clone, Copy, Display)]
#[display(fmt = "LeakyReLU {}", leak_rate)]
pub(super) struct LeakyReLU {
    pub leak_rate: f64,
}

impl SimpleActivationFunction for LeakyReLU {
    #[inline]
    fn propagate(&self, input: f64) -> f64 {
        leaky_relu(input, self.leak_rate)
    }

    #[inline]
    fn derivative_from_output(&self, self_output: f64) -> f64 {
        if self_output.is_sign_positive() { 1.0 } else { self.leak_rate }
    }
}

#[inline]
fn leaky_relu(x: f64, leak_rate: f64) -> f64 {
    if x.is_sign_positive() { x } else { leak_rate * x }
}
