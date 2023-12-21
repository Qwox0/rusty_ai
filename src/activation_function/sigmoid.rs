use super::SimpleActivationFunction;
use derive_more::Display;
use std::ops::{Add, Neg};

#[derive(Debug, Clone, Copy, Display)]
pub(super) struct Sigmoid;

impl SimpleActivationFunction for Sigmoid {
    fn propagate(&self, input: f64) -> f64 {
        input.neg().exp().add(1.0).recip()
    }

    fn derivative_from_output(&self, self_output: f64) -> f64 {
        self_output * (1.0 - self_output)
    }
}
