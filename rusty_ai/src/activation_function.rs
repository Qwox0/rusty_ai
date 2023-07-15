use crate::prelude::*;
use std::ops::{Add, Neg};

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

    /// Softmax(X) : x_i -> e^x_i/sum e^x for x in X
    Softmax,
    ///// LogSoftmax(X) = ln(Softmax(X)) = ln( e^x/sum e^x )
    //LogSoftmax,
}

#[inline]
fn relu(x: f64) -> f64 {
    leaky_relu(x, 0.0)
}

#[inline]
fn leaky_relu(x: f64, leak_rate: f64) -> f64 {
    if x.is_sign_positive() { x } else { leak_rate * x }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    x.neg().exp().add(1.0).recip()
}

fn softmax(vec: &mut Vec<f64>) {
    vec.iter_mut().assign_each(f64::exp);
    let sum: f64 = vec.iter().sum();
    vec.iter_mut().assign_each(|x| x / sum);
}

impl ActivationFn {
    pub const fn default_relu() -> ActivationFn {
        ActivationFn::ReLU(1.0)
    }

    pub const fn default_leaky_relu() -> ActivationFn {
        ActivationFn::LeakyReLU(0.01, 1.0)
    }

    pub fn calc_mut(&self, buf: &mut Vec<f64>) {
        match self {
            ActivationFn::Identity => (),
            ActivationFn::ReLU(_) => buf.iter_mut().assign_each(relu),
            ActivationFn::LeakyReLU(leak_rate, _) => {
                buf.iter_mut().assign_each(|x| leaky_relu(x, *leak_rate))
            },
            ActivationFn::Sigmoid => buf.iter_mut().assign_each(sigmoid),
            ActivationFn::Softmax => todo!(),
        }
    }

    pub fn calculate(&self, mut input: Vec<f64>) -> Vec<f64> {
        self.calc_mut(&mut input);
        input
    }

    pub fn calculate2(&self, mut input: Vec<f64>) -> Vec<f64> {
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
            ReLU(_) => relu(input),
            LeakyReLU(leak_rate, _) => leaky_relu(input, *leak_rate),
            Sigmoid => sigmoid(input),
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
            Softmax => panic!("Softmax needs entire Vector"),
        }
    }

    pub fn derivative_from_output(&self, output: &Vec<f64>) -> Vec<f64> {
        todo!()
    }

    fn call_single(&self, input: (f64,)) -> f64 {
        todo!()
    }

    fn call_vec(&self, input: (Vec<f64>,)) -> Vec<f64> {
        todo!()
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
        }
    }
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

    #[test]
    fn test_relu() {
        const d0: f64 = 0.5;
        const pi: f64 = std::f64::consts::PI;
        let relu = ActivationFn::ReLU(d0);

        assert_eq!(relu._calculate_single(-10.0), 0.0);
        assert_eq!(relu._calculate_single(0.0), 0.0);
        assert_eq!(relu._calculate_single(1.0), 1.0);
        assert_eq!(relu._calculate_single(pi), pi);
        assert_eq!(relu._calculate_single(10.0), 10.0);

        assert_eq!(relu.derivative(-10.0), 0.0);
        assert_eq!(relu.derivative(0.0), d0);
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(pi), 1.0);
        assert_eq!(relu.derivative(10.0), 1.0);
    }
}

mod benches {
    extern crate test;

    use super::*;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_relu(b: &mut Bencher) {
        let relu = ActivationFn::ReLU(0.5);
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();

        b.iter(|| black_box(relu.calc_mut(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn bench_relu_match(b: &mut Bencher) {
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();
        fn relu(vec: &mut Vec<f64>) {
            vec.iter_mut().for_each(|x| {
                *x = match *x {
                    x if x.is_sign_positive() => x,
                    _ => 0.0,
                }
            })
        }

        b.iter(|| black_box(relu(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn bench_relu_match_copy(b: &mut Bencher) {
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();
        fn relu_single(input: f64) -> f64 {
            match input {
                x if x.is_sign_positive() => x,
                _ => 0.0,
            }
        }
        fn relu(vec: &mut Vec<f64>) {
            vec.iter_mut().for_each(|x| *x = relu_single(*x))
        }

        b.iter(|| black_box(relu(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn bench_relu_match_assign_each(b: &mut Bencher) {
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();
        fn relu_single(input: f64) -> f64 {
            match input {
                x if x.is_sign_positive() => x,
                _ => 0.0,
            }
        }
        fn relu(vec: &mut Vec<f64>) {
            vec.iter_mut().assign_each(relu_single)
        }

        b.iter(|| black_box(relu(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn bench_relu_match_mul(b: &mut Bencher) {
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();
        fn relu(vec: &mut Vec<f64>) {
            vec.iter_mut().for_each(|x| {
                *x *= match *x {
                    x if x.is_sign_positive() => 1.0,
                    _ => 0.0,
                }
            })
        }

        b.iter(|| black_box(relu(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn bench_relu_branchless(b: &mut Bencher) {
        let mut nums: Vec<_> = (-100..100).map(f64::from).collect();
        fn relu(vec: &mut Vec<f64>) {
            vec.iter_mut().for_each(|x| *x *= x.is_sign_positive() as u64 as f64)
        }

        b.iter(|| black_box(relu(black_box(&mut nums.clone()))))
    }

    #[bench]
    fn calculate_relu(b: &mut Bencher) {
        let relu = ActivationFn::ReLU(0.5);
        let nums: Vec<_> = (-100..100).map(f64::from).collect();

        b.iter(|| black_box(relu.calculate(black_box(nums.clone()))))
    }

    #[bench]
    fn calculate2_relu(b: &mut Bencher) {
        let relu = ActivationFn::ReLU(0.5);
        let nums: Vec<_> = (-100..100).map(f64::from).collect();

        b.iter(|| black_box(relu.calculate2(black_box(nums.clone()))))
    }
}
