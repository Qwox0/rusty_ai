use crate::util::impl_fn_traits;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, Copy, Default)]
pub enum ActivationFn {
    /// Identity(x) = x
    /// Identity(x) = 1
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

    /// Softmax(x) = e^x/sum e^x
    Softmax,
}

impl ActivationFn {
    pub const fn default_relu() -> ActivationFn { ActivationFn::ReLU(1.0) }

    pub const fn default_leaky_relu() -> ActivationFn { ActivationFn::LeakyReLU(0.01, 1.0) }

    pub fn calculate(&self, mut input: Vec<f64>) -> Vec<f64> {
        match self {
            ActivationFn::Softmax => softmax(&mut input),
            act_fn => input.iter_mut().for_each(|i| {
                *i = act_fn._calculate_single(*i);
            }),
        }
        input
    }

    fn _calculate_single(&self, input: f64) -> f64 {
        use ActivationFn::*;
        match self {
            Identity => input,
            ReLU(_) => match input {
                x if x.is_sign_positive() => x,
                _ => 0.0,
            },
            LeakyReLU(leak_rate, _) => match input {
                x if x.is_sign_positive() => x,
                x => leak_rate * x,
            },
            Sigmoid => 1.0 / (1.0 + f64::exp(-input)),
            Softmax => panic!("Softmax needs entire Vector"),
        }
    }

    pub fn derivative(&self, input: f64) -> f64 {
        use ActivationFn::*;
        #[allow(illegal_floating_point_literal_pattern)] // for pattern: 0.0 => ...
        match self {
            Identity => 1.0,
            ReLU(d0) => match input {
                0.0 => *d0,
                x => x.is_sign_positive() as u8 as f64,
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
            Softmax => panic!("Softmax needs entire Vector"),
        }
    }

    pub fn derivative_from_output(&self, output: &Vec<f64>) -> Vec<f64> {
        todo!()
    }
}

impl Fn<(Vec<f64>,)> for ActivationFn {
    extern "rust-call" fn call(&self, args: (Vec<f64>,)) -> Self::Output {
        self.calculate(args.0)
    }
}

impl FnMut<(Vec<f64>,)> for ActivationFn {
    extern "rust-call" fn call_mut(&mut self, args: (Vec<f64>,)) -> Self::Output { self.call(args) }
}

impl FnOnce<(Vec<f64>,)> for ActivationFn {
    type Output = Vec<f64>;

    extern "rust-call" fn call_once(mut self, args: (Vec<f64>,)) -> Self::Output {
        self.call_mut(args)
    }
}

impl std::fmt::Display for ActivationFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ActivationFn::*;
        match self {
            Identity => write!(f, "Identity"),
            ReLU(d0) => write!(f, "ReLU (ReLU'(0)={})", d0),
            LeakyReLU(a, d0) => write!(f, "Leaky ReLU (a={}; f'(0)={})", a, d0),
            Sigmoid => write!(f, "Sigmoid"),
            Softmax => write!(f, "Softmax"),
        }
    }
}

fn softmax(vec: &mut Vec<f64>) {
    vec.iter_mut().for_each(|x| *x = x.exp());
    let sum: f64 = vec.iter().sum();
    vec.iter_mut().for_each(|x| *x /= sum);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_wiki() {
        let mut data = vec![1, 2, 3, 4, 1, 2, 3].into_iter().map(f64::from).collect();
        softmax(&mut data);
        println!("{:?}", data);
        let rounded: Vec<_> = data.iter().map(|x| (1000.0 * x).round() / 1000.0).collect();
        println!("{:?}", rounded);
        assert_eq!(rounded, &[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]);
    }
}
