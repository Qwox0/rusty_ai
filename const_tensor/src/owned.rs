use crate::{
    data, maybe_uninit::MaybeUninit, multidim_arr::MultidimArr,
    multidimensional::MultidimensionalOwned, tensor, Element, Len, Shape, Vector,
};
use serde::{Deserialize, Serialize};
use std::{
    borrow::{Borrow, BorrowMut},
    fmt, mem,
    ops::{Deref, DerefMut},
};

/// An owned tensor.
///
/// # `serde` Note
///
/// Tensors are serialized in their 1D representation. This means, that serialization followed by
/// deserialization is equivalent to [`tensor::transmute_clone`].
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

    /// similar to [`Vec::from_raw_parts`].
    pub unsafe fn from_raw_ptr(ptr: *mut X) -> Self {
        let ptr = ptr as *mut tensor<X, S>;
        unsafe { Box::from_raw(ptr) }.into()
    }

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

pub struct InvalidLenError {
    expected: usize,
    got: usize,
}

impl<X: Element, S: Shape> TryFrom<Box<[X]>> for Tensor<X, S> {
    type Error = InvalidLenError;

    fn try_from(data: Box<[X]>) -> Result<Self, Self::Error> {
        if data.len() != S::LEN {
            return Err(InvalidLenError { expected: S::LEN, got: data.len() });
        }
        let ptr = Box::into_raw(data) as *mut X;
        Ok(unsafe { Self::from_raw_ptr(ptr) })
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

impl<X: Element + Serialize, S: Shape> Serialize for Tensor<X, S> {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where Ser: serde::Serializer {
        tensor::<X, S>::serialize(&self, serializer)
    }
}

impl<'de, X: Element + Deserialize<'de>, S: Shape> Deserialize<'de> for Tensor<X, S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        Box::<[X]>::deserialize(deserializer)?.try_into().map_err(
            |InvalidLenError { expected, got }| {
                serde::de::Error::custom(format_args!(
                    "invalid length: {}, expected {}",
                    got, expected
                ))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix, Multidimensional};

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
