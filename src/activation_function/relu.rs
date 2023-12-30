use crate::{gradient::aliases::WeightedSumGradient, ActivationFunction};
use derive_more::Display;
use matrix::Num;

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct ReLU;

impl<X: Num> ActivationFunction<X> for ReLU {
    #[inline]
    fn propagate(&self, input: Vec<X>) -> Vec<X> {
        input.into_iter().map(|x| leaky_relu(x, X::lit(0))).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: crate::gradient::aliases::OutputGradient<X>,
        self_output: &[X],
    ) -> WeightedSumGradient<X> {
        output_gradient
            .into_iter()
            .zip(self_output)
            .map(|(dl_dy, y)| dl_dy * if y.is_positive() { X::lit(1) } else { X::lit(0) })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Display)]
#[display(fmt = "LeakyReLU {}", leak_rate)]
pub(super) struct LeakyReLU<X> {
    pub leak_rate: X,
}

impl<X: Num> ActivationFunction<X> for LeakyReLU<X> {
    #[inline]
    fn propagate(&self, input: Vec<X>) -> Vec<X> {
        input.into_iter().map(|x| leaky_relu(x, self.leak_rate)).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: crate::gradient::aliases::OutputGradient<X>,
        self_output: &[X],
    ) -> WeightedSumGradient<X> {
        output_gradient
            .into_iter()
            .zip(self_output)
            .map(|(dl_dy, y)| dl_dy * if y.is_positive() { X::lit(1) } else { self.leak_rate })
            .collect()
    }
}

#[inline]
fn leaky_relu<X: Num>(x: X, leak_rate: X) -> X {
    if x.is_positive() { x } else { leak_rate * x }
}
