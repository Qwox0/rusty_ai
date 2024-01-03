/*
use crate::{private, tensor, Element, Shape, Tensor};
use std::{mem, ops::Deref};

/// Tensor trait. This is implemented by [`Tensor`].
///
/// implemented for `Tensor<[[[[X; A]; B]; ...]; Z]>` (any dimensions)
///
/// `SUB`: sub tensor
/// `LEN`: 1D data length
pub trait TensorI<X: Element, const LEN: usize>:
    private::Sealed<X>
    + Sized
    + From<Box<tensor<Self::Shape, LEN>>>
    + AsRef<tensor<Self::Shape, LEN>>
    + AsMut<tensor<Self::Shape, LEN>>
    + Deref<Target = tensor<Self::Shape, LEN>>
{
    /// assert!(Shape::LEN == LEN)
    type Shape: Shape;

    /// Returns the inner representation of the tensor.
    fn into_inner(self) -> Box<tensor<Self::Shape, LEN>>;

    /// Creates the Tensor from a 1D representation of its elements.
    fn from_1d(vec: Tensor<[X; Self::Shape::LEN], { Self::Shape::LEN }>) -> Self;
}
*/
