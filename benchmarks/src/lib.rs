#![allow(unused)]
#![feature(test)]

extern crate test;

mod macros;
pub(crate) use macros::make_benches;

pub(crate) mod constants;

mod matrix;
mod layer;
