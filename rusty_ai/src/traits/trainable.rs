use crate::prelude::*;

pub trait Trainable {
    fn init_zero_gradient(&self) -> Gradient;
}
