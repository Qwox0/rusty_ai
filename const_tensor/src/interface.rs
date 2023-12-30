use crate::{private, Element, Tensor, TensorData};
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
    + From<Box<Self::Data>>
    + AsRef<Self::Data>
    + AsMut<Self::Data>
    + Deref<Target = Self::Data>
{
    /// Tensor data
    type Data: TensorData<Element = X>;

    /// Returns the inner representation of the tensor.
    fn into_inner(self) -> Box<Self::Data>;

    /// Creates the Tensor from a 1D representation of its elements.
    fn from_1d(vec: Tensor<[X; LEN]>) -> Self {
        Self::from(unsafe { mem::transmute(vec) })
    }
}
