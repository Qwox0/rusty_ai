use crate::{
    data,
    maybe_uninit::MaybeUninit,
    multidim_arr::MultidimArr,
    multidimensional::{Multidimensional, MultidimensionalOwned},
    tensor, Element, Float, Len, Num, Shape, Vector,
};
use serde::{Deserialize, Serialize};
use std::{
    alloc,
    borrow::{Borrow, BorrowMut},
    fmt,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr,
};

/// An owned tensor.
#[repr(transparent)]
pub struct Tensor<X: Element, S: Shape> {
    /// [`Box`] pointer to the tensor data.
    pub(crate) data: Box<tensor<X, S>>,
}

impl<X: Element, S: Shape> Tensor<X, S> {
    /// Creates a new [`Tensor`] on the heap.
    #[inline]
    pub fn new(data: S::Mapped<X>) -> Self {
        Self::from(data::tensor::new_boxed(data))
    }

    /*
    /// Allocates a new uninitialized [`Tensor`].
    #[inline]
    pub fn new_uninit() -> Tensor<MaybeUninit<X>, S> {
        Tensor::from(tensor::<X, S>::new_boxed_uninit())
    }

    /// Creates a new Tensor filled with the values in `iter`.
    /// If the [`Iterator`] it too small, the rest of the elements contain the value `X::default()`.
    #[inline]
    pub fn from_iter(iter: impl IntoIterator<Item = X>) -> Self {
        let mut t = Self::default();
        t.iter_elem_mut().zip(iter).for_each(|(x, val)| *x = val);
        t
    }
    */

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    pub fn from_1d<const LEN: usize>(vec: Vector<X, LEN>) -> Self {
        let vec = mem::ManuallyDrop::new(vec);
        unsafe { mem::transmute_copy(&vec) }
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    pub fn into_1d<const LEN: usize>(self) -> Vector<X, LEN>
    where S: Len<LEN> {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    pub fn transmute_into<S2, const LEN: usize>(self) -> Tensor<X, S2>
    where
        S: Len<LEN>,
        S2: Shape + Len<LEN>,
    {
        let t = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&t) }
    }
}

impl<X: Element, S: Shape> MultidimensionalOwned<X> for Tensor<X, S> {
    type Data = tensor<X, S>;
    type Uninit = Tensor<MaybeUninit<X>, S>;

    fn new_uninit() -> Self::Uninit {
        Tensor::from(tensor::<X, S>::new_boxed_uninit())
    }
}

impl<X: Element, S: Shape> MultidimensionalOwned<X> for Box<tensor<X, S>> {
    type Data = tensor<X, S>;
    type Uninit = Box<tensor<MaybeUninit<X>, S>>;

    fn new_uninit() -> Self::Uninit {
        tensor::<X, S>::new_boxed_uninit()
    }
}

impl<X: Element + fmt::Debug, S: Shape> fmt::Debug for Tensor<X, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tensor").field(&self.0).finish()
    }
}

impl<X: Element + Clone, S: Shape> Clone for Tensor<X, S> {
    fn clone(&self) -> Self {
        self.data.to_owned()
    }
}

impl<X: Element, S: Shape> Default for Tensor<X, S> {
    fn default() -> Self {
        tensor::<X, S>::default().to_owned()
    }
}

impl<X: Element, S: Shape> PartialEq<Self> for Tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<X: Element, S: Shape> Eq for Tensor<X, S> where S::Mapped<X>: Eq {}

impl<X: Element, S: Shape> PartialEq<&tensor<X, S>> for Tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &&tensor<X, S>) -> bool {
        self.as_ref() == *other
    }
}

impl<X: Element, S: Shape + PartialEq> PartialEq<Tensor<X, S>> for &tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &Tensor<X, S>) -> bool {
        *self == other.as_ref()
    }
}

impl<X: Element, S: Shape, D: MultidimArr<Element = X, Mapped<()> = S>> PartialEq<D>
    for Tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &D) -> bool {
        self.0 == other.type_hint()
    }
}

impl<X: Element, S: Shape> From<Box<data::tensor<X, S>>> for Tensor<X, S> {
    fn from(data: Box<data::tensor<X, S>>) -> Self {
        Tensor { data }
    }
}

impl<X: Element, S: Shape> From<Tensor<X, S>> for Box<data::tensor<X, S>> {
    fn from(t: Tensor<X, S>) -> Self {
        t.data
    }
}

impl<X: Element, S: Shape> Deref for Tensor<X, S> {
    type Target = data::tensor<X, S>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<X: Element, S: Shape> DerefMut for Tensor<X, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<X: Element, S: Shape> AsRef<data::tensor<X, S>> for Tensor<X, S> {
    fn as_ref(&self) -> &data::tensor<X, S> {
        &self.data
    }
}

impl<X: Element, S: Shape> AsMut<data::tensor<X, S>> for Tensor<X, S> {
    fn as_mut(&mut self) -> &mut data::tensor<X, S> {
        &mut self.data
    }
}

impl<X: Element, S: Shape> Borrow<data::tensor<X, S>> for Tensor<X, S> {
    fn borrow(&self) -> &data::tensor<X, S> {
        &self.data
    }
}

impl<X: Element, S: Shape> BorrowMut<data::tensor<X, S>> for Tensor<X, S> {
    fn borrow_mut(&mut self) -> &mut data::tensor<X, S> {
        &mut self.data
    }
}

impl<X: Element + Serialize, S: Shape> Serialize for Tensor<X, S>
where <S::Mapped<X> as MultidimArr>::Wrapped: Serialize
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where Ser: serde::Serializer {
        tensor::<X, S>::serialize(&self, serializer)
    }
}

impl<'de, X: Element + Deserialize<'de>, S: Shape> Deserialize<'de> for Tensor<X, S>
where <S::Mapped<X> as MultidimArr>::Wrapped: Deserialize<'de>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        <S::Mapped<X> as MultidimArr>::Wrapped::deserialize(deserializer)
            .map(S::Mapped::unwrap)
            .map(Tensor::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn safety_new_uninit() {
        type Mat = Matrix<i32, 3, 2>;
        let mut t = Mat::new_uninit();
        for (idx, x) in t.iter_elem_mut().enumerate() {
            x.write(idx as i32);
        }
        let t: Mat = unsafe { mem::transmute(t) };
        assert_eq!(t, tensor::literal([[0, 1, 2], [3, 4, 5]]));
    }
}
