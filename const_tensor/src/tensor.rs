use crate::{Element, Float, Len, Num, TensorData, Vector};
use core::fmt;
use std::{
    borrow::{Borrow, BorrowMut},
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
    + Clone
    + Default
    + fmt::Debug
    + Deref<Target = Self::Data>
    + DerefMut
    + AsRef<Self::Data>
    + AsMut<Self::Data>
    + Borrow<Self::Data>
    + BorrowMut<Self::Data>
{
    /// The tensor data which represents the data type on the heap.
    type Data: TensorData<X, Owned = Self>;

    /// `Self` but with another [`Element`] type.
    type Mapped<E: Element>: Tensor<E, Data = <Self::Data as TensorData<X>>::Mapped<E>>;

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
    fn transmute_into<U2: Tensor<X>, const LEN: usize>(self) -> U2
    where
        Self::Data: Len<LEN>,
        U2::Data: Len<LEN>,
    {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }

    /// Applies a function to every element of the tensor.
    // TODO: bench vs TensorData::map_clone
    #[inline]
    fn map_elem<const LEN: usize>(mut self, mut f: impl FnMut(X) -> X) -> Self
    where Self::Data: Len<LEN> {
        self.map_elem_mut(|x| *x = f(*x));
        self
    }

    /// Multiplies the tensor by a scalar value.
    #[inline]
    fn scalar_mul<const LEN: usize>(mut self, scalar: X) -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        self.scalar_mul_mut(scalar);
        self
    }

    /// Adds `other` to `self` elementwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[1, 2], [3, 4]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// let res = mat1.add_elem(&mat2);
    /// assert_eq!(res._as_inner(), &[[5, 5], [5, 5]]);
    /// ```
    #[inline]
    fn add_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        self.add_elem_mut(other);
        self
    }

    /// Subtracts `other` from `self` elementwise.
    #[inline]
    fn sub_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        self.sub_elem_mut(other);
        self
    }

    /// Multiplies `other` to `self` elementwise.
    #[inline]
    fn mul_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        Self::Data: Len<LEN>,
        X: Num,
    {
        self.mul_elem_mut(other);
        self
    }

    /// Calculates the reciprocal of every element in `self`.
    #[inline]
    fn recip_elem<const LEN: usize>(mut self) -> Self
    where
        Self::Data: Len<LEN>,
        X: Float,
    {
        self.recip_elem_mut();
        self
    }

    /// Calculates the negative of the tensor.
    #[inline]
    fn neg<const LEN: usize>(mut self) -> Self
    where
        Self::Data: Len<LEN>,
        X: Float,
    {
        self.neg_mut();
        self
    }
}
