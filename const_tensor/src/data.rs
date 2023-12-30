use crate::{private, Element, TensorI};
use core::slice;
use std::{mem, ops::Deref};

/// is implemented for `X` and `[X; N]` where `X` implements [`Element`] and `const N: usize`.
pub trait TensorData: private::SealedData + Sized {
    /// Represents a single value in the tensor, like a 0D Tensor.
    type Element: Element;

    /// one dimension lower
    type SubData: TensorData<Element = Self::Element>;

    /// dimension of the tensor data.
    const DIM: usize;

    /// total count of elements in the tensor data.
    const LEN: usize;

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d(&self) -> &[Self::Element; Self::LEN] {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d_mut(&mut self) -> &mut [Self::Element; Self::LEN] {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    fn transmute_as<T: TensorI<Self::Element, { Self::LEN }>>(&self) -> &<T as Deref>::Target {
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    fn iter_sub_tensors<'a>(&'a self) -> slice::Iter<'a, Self::SubData>;

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    fn iter_sub_tensors_mut<'a>(&'a mut self) -> slice::IterMut<'a, Self::SubData>;

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    fn iter_elem(&self) -> slice::Iter<'_, Self::Element>
    where [Self::Element; Self::LEN]: Sized {
        self.as_1d().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    #[inline]
    fn iter_elem_mut(&mut self) -> slice::IterMut<'_, Self::Element>
    where [Self::Element; Self::LEN]: Sized {
        self.as_1d_mut().iter_mut()
    }
}

impl<X: Element> private::SealedData for X {}
impl<X: Element> TensorData for X {
    type Element = X;
    type SubData = X;

    const DIM: usize = 1;
    const LEN: usize = 1;

    fn iter_sub_tensors(&self) -> slice::Iter<'_, X> {
        unsafe { mem::transmute::<&Self, &[X; 1]>(self) }.iter()
    }

    fn iter_sub_tensors_mut(&mut self) -> slice::IterMut<'_, X> {
        unsafe { mem::transmute::<&mut Self, &mut [X; 1]>(self) }.iter_mut()
    }
}

impl<T: TensorData, const N: usize> private::SealedData for [T; N] {}
impl<X: Element, T: TensorData<Element = X>, const N: usize> TensorData for [T; N] {
    type Element = X;
    type SubData = T;

    const DIM: usize = T::DIM + 1;
    const LEN: usize = T::LEN * N;

    fn iter_sub_tensors(&self) -> slice::Iter<'_, T> {
        self.iter()
    }

    fn iter_sub_tensors_mut(&mut self) -> slice::IterMut<'_, T> {
        self.iter_mut()
    }
}

/// Type alias for a 0D tensor.
pub type ScalarData<X> = X;
/// Type alias for a 1D tensor.
pub type VectorData<X, const LEN: usize> = [X; LEN];
/// Type alias for a 2D tensor.
pub type MatrixData<X, const W: usize, const H: usize> = [[X; W]; H];
/// Type alias for a 3D tensor.
pub type Tensor3Data<X, const A: usize, const B: usize, const C: usize> = [[[X; A]; B]; C];
/// Type alias for a 4D tensor.
pub type Tensor4Data<X, const A: usize, const B: usize, const C: usize, const D: usize> =
    [[[[X; A]; B]; C]; D];
