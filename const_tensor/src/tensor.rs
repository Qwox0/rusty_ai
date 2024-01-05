use crate::{Element, Len, Num, TensorData, Vector};
use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
    mem,
    ops::{Deref, DerefMut},
};

/// Trait for Owned Tensor types.
///
/// # SAFETY
///
/// The data structure should equal [`Box<Self::Data>`] as [`mem::transmute`] and
/// [`mem::transmute_copy`] are used to convert between [`Tensor`] types.
pub unsafe trait Tensor<X: Element>:
    Sized
    + Debug
    + Deref<Target = Self::Data>
    + DerefMut
    + AsRef<Self::Data>
    + AsMut<Self::Data>
    + Borrow<Self::Data>
    + BorrowMut<Self::Data>
{
    /// The tensor data which represents the data type on the heap.
    type Data: TensorData<X, Owned = Self>;

    /// Creates a new Tensor.
    fn from_box(data: Box<Self::Data>) -> Self;

    /// Creates a new Tensor.
    #[inline]
    fn new(data: <Self::Data as TensorData<X>>::Shape) -> Self {
        Self::from_box(Self::Data::new_boxed(data))
    }

    /// Creates a new Tensor filled with the values in `iter`.
    /// If the [`Iterator`] it too small, the rest of the elements contain the scalar value `0`.
    #[inline]
    fn from_iter<const LEN: usize>(iter: impl IntoIterator<Item = X>) -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        let mut tensor = Self::zeros();
        tensor.iter_elem_mut().zip(iter).for_each(|(x, val)| *x = val);
        tensor
    }

    /// Creates a new Tensor filled with the scalar value.
    #[inline]
    fn full<const LEN: usize>(val: X) -> Self
    where Self::Data: Len<LEN> {
        Self::from_1d(Vector::new([val; LEN]))
    }

    /// Creates a new Tensor filled with the scalar value `0`.
    #[inline]
    fn zeros<const LEN: usize>() -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        Self::full(X::ZERO)
    }

    /// Creates a new Tensor filled with the scalar value `1`.
    #[inline]
    fn ones<const LEN: usize>() -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        Self::full(X::ONE)
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    fn from_1d<const LEN: usize>(vec: Vector<X, LEN>) -> Self
    where Self::Data: Len<LEN> {
        let vec = mem::ManuallyDrop::new(vec);
        unsafe { mem::transmute_copy(&vec) }
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    fn into_1d<const LEN: usize>(self) -> Vector<X, LEN>
    where Self::Data: Len<LEN> {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    fn transmute_into<T2: Tensor<X>, const LEN: usize>(self) -> T2
    where
        Self::Data: Len<LEN>,
        T2::Data: Len<LEN>,
    {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem_mut<const LEN: usize>(mut self, f: impl FnMut(&mut X)) -> Self
    where Self::Data: Len<LEN> {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem<const LEN: usize>(self, mut f: impl FnMut(X) -> X) -> Self
    where Self::Data: Len<LEN> {
        self.map_elem_mut(|x| *x = f(*x))
    }
}
