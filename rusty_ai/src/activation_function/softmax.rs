use crate::prelude::*;
use derive_more::Display;

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct Softmax;

impl ActivationFunction for Softmax {
    fn propagate(&self, mut input: Vec<f64>) -> Vec<f64> {
        let mut sum = 0.0;
        for x in input.iter_mut() {
            *x = x.exp();
            sum += *x;
        }
        input.into_iter().map(|x| x / sum).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
        // dL/dx_i = y_i * (dL/dy_i - sum dL/dy_k * y_k for k in 1..=n)
        let s: f64 = output_gradient.iter().zip(self_output).map(|(&dl_dy, &y)| dl_dy * y).sum();
        // dL/dx_i = y_i * (dL/dy_i - s)
        output_gradient
            .into_iter()
            .map(|out_grad| out_grad - s)
            .zip(self_output)
            .map(|(out_grad, out)| out * out_grad)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct LogSoftmax;

impl ActivationFunction for LogSoftmax {
    fn propagate(&self, input: Vec<f64>) -> Vec<f64> {
        // ln(e^(x_i)/exp_sum) == x_i - ln(exp_sum)
        let ln_sum = input.iter().copied().map(f64::exp).sum::<f64>().ln();
        input.into_iter().map(|x| x - ln_sum).collect()
    }

    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
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
    }
}
