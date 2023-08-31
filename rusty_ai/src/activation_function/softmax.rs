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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_backprop() {
        let log_softmax = LogSoftmax;
        let nllloss = NLLLoss;

        let input = [0.0, 0.5, 3.0];
        let expected = 3;
        let expected = &(expected - 1);

        println!("x: {:?}", input);

        let o = log_softmax.propagate(input.to_vec());
        let o: [f64; 3] = o.as_slice().try_into().unwrap();

        println!("o: {:?}", o);

        let loss = nllloss.propagate_arr(&o, expected);

        println!("loss: {:?}", loss);

        let d_o = nllloss.backpropagate_arr(&o, expected);

        println!("d_o: {:?}", d_o);
        assert_eq!(d_o, vec![0.0, 0.0, -1.0]);

        let d_x = log_softmax.backpropagate(d_o, o.as_slice());

        println!("d_x: {:?}", d_x);
        assert_eq!(d_x, vec![0.04398648, 0.072521446, -0.11650793]);
    }

    #[test]
    fn simple_log_softmax() {
        let log_softmax = LogSoftmax;

        let out = log_softmax.propagate(vec![2.0, 0.0, 0.35]);

        println!("out: {:?}", out);

        let nllloss = NLLLoss;

        let out: Result<[f64; 3], _> = out.as_slice().try_into();
        let err = nllloss.propagate_arr(&out.unwrap(), &0);

        println!("err: {:?}", err);

        panic!()
    }

    #[test]
    fn log_softmax_prop() {
        let input = [0.01, 0.999, 0.5];
        let log_softmax = LogSoftmax;

        let out = log_softmax.propagate(input.to_vec());

        println!("out: {:?}", out);

        let nllloss = NLLLoss;

        let out: Result<[f64; 3], _> = out.as_slice().try_into();
        let err = nllloss.propagate_arr(&out.unwrap(), &0);

        println!("err: {:?}", err);

        panic!()
    }
}
