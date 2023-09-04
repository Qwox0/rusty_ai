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

    ///
    /// 0.0   o1
    /// 0.1   o2    L
    /// 3.0   o3
    ///
    /// s = sum of e^x for x in X = e^1 + e^2 + e^3 = 30.1928748505773
    /// ln s = 3.407605964444
    ///
    /// o = ln( e^x/s ) = x - ln s
    /// o1 = 0 - ln s   = -3.407605964444
    /// o2 = 0.1 - ln s = -3.307605964444
    /// o3 = 3 - ln s   = -0.407605964444
    ///
    /// pred. change = e^o
    /// c1 = 0.033120396946
    /// c2 = 0.0366036995001
    /// c3 = 0.665240955775
    ///
    /// let expected out = 3
    ///
    /// L = -o3 = 0.407605964444
    ///
    /// dL/do1 = 0
    /// dL/do2 = 0
    /// dL/do3 = -1
    ///
    /// o3 = x3 - ln(s)
    ///
    /// do3/dx1 = -1/s * e^x1    = -softmax(x1)  = 0.0331203969462
    /// do3/dx2 = -1/s * e^x2    = -softmax(x2)  =
    /// do3/dx3 = 1 - 1/s * e^x3 = 1-softmax(x3) =
    ///
    /// dL/x1 = dL/do1 * do1/x1 + dL/do2 * do2/x1 + dL/do3 * do3/x1
    /// dL/x1 =      0 * do1/x1 +      0 * do2/x1 -      1 * do3/x1
    /// dL/x1 = -1 * do3/x1 = softmax(x1)
    /// dL/x2 = -1 * do3/x2 = softmax(x2)
    /// dL/x3 = -1 * do3/x3 = softmax(x3) - 1
    fn backpropagate(
        &self,
        output_gradient: OutputGradient,
        self_output: &[f64],
    ) -> WeightedSumGradient {
        // dL/dx_i = dL/dy_i - sum dL/dy_k * exp(y_k) for k in 1..=n
        // println!("self_output: {:?}", self_output);
        // println!("output_gradient: {:?}", output_gradient);
        // let s: f64 = self_output
        //     .iter()
        //     .copied()
        //     .map(f64::exp)
        //     .zip(&output_gradient)
        //     .map(|(exp_y, dl_dy)| dl_dy * exp_y)
        //     .sum();
        // // dL/dx_i = dL/dy_i - s
        // output_gradient.into_iter().map(|dl_dy| dl_dy - s).collect()

        //
        self_output
            .iter()
            .copied()
            .map(f64::exp)
            .zip(&output_gradient)
            .map(|(exp_y, dl_dy)| exp_y + dl_dy)
            .collect()
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
        let expected_d_x = vec![0.0439864802921, 0.0725214456807, -0.116507925873];
        let diff = d_x.iter().zip(expected_d_x).map(|(a, b)| a - b).sum::<f64>();
        println!("diff: {diff}");
        assert!(diff.abs() < 1e-10);
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
