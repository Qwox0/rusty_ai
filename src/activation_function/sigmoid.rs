use crate::{
    gradient::aliases::{OutputGradient, WeightedSumGradient},
    ActivationFunction,
};
use derive_more::Display;
use matrix::Float;

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct Sigmoid;

impl<X: Float> ActivationFunction<X> for Sigmoid {
    fn propagate(&self, input: Vec<X>) -> Vec<X> {
        input.into_iter().map(|x| x.neg().exp().add(X::lit(1)).recip()).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: OutputGradient<X>,
        self_output: &[X],
    ) -> WeightedSumGradient<X> {
        output_gradient
            .into_iter()
            .zip(self_output)
            .map(|(dl_dy, &y)| y * (X::lit(1) - y) * dl_dy)
            .collect()
    }
}
