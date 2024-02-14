use crate::{
    maybe_uninit::MaybeUninit,
    multidim_arr::{Len, MultidimArr},
    multidimensional::{Multidimensional, MultidimensionalOwned},
    owned::Tensor,
    scalar, vector, Element, Shape,
};
use core::mem;
use serde::{ser::SerializeSeq, Serialize};
use std::{
    alloc,
    iter::Map,
    ops::{Index, IndexMut},
    ptr, slice,
};

/// Data of a [`Tensor`].
///
/// Like unsized type, this type should never be owned directly. Thus this type is similar to `str`
/// and slice.
///
/// # `serde` Note
///
/// Tensors are serialized in their 1D representation. This means, that serialization followed by
/// deserialization is equivalent to [`tensor::transmute_clone`]:
///
/// ```rust
/// # use const_tensor::*;
/// # fn main() -> Result<(), serde_json::Error> {
/// let tensor = matrix::literal([[1, 2, 3, 4], [5, 6, 7, 8]]); // 2D
/// let json = serde_json::to_string(tensor)?;
/// assert_eq!(json, "[1,2,3,4,5,6,7,8]"); // 1D
/// let deserialized: Tensor3<i32, 2, 2, 2> = serde_json::from_str(&json)?;
/// assert_eq!(deserialized, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // 3D
///
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct tensor<X: Element, S: Shape>(
    /// inner tensor data value
    pub(crate) S::Mapped<X>,
);

impl<X: Element, S: Shape> tensor<X, S> {
    /// a tensor must be allocated on the heap -> use `new_boxed` or [`Tensor`].
    #[inline]
    pub(crate) fn new(data: S::Mapped<X>) -> Self {
        Self(data)
    }

    /// similar to [`slice::from_raw_parts`].
    pub unsafe fn from_raw_ptr<'a>(ptr: *const X) -> &'a Self {
        let ref_ = unsafe { ptr.as_ref().unwrap_unchecked() };
        Self::wrap_ref(unsafe { mem::transmute(ref_) })
    }

    /// Allocates a new tensor on the heap.
    ///
    /// You should probably use the [`Tensor`] wrapper instead.
    #[inline]
    pub fn new_boxed(data: S::Mapped<X>) -> Box<Self> {
        Box::new(Self::new(data))
    }

    #[inline]
    pub(crate) fn new_boxed_uninit() -> Box<tensor<MaybeUninit<X>, S>> {
        let ptr = if mem::size_of::<X>() == 0 {
            ptr::NonNull::dangling()
        } else {
            let layout = alloc::Layout::new::<tensor<MaybeUninit<X>, S>>();
            // SAFETY: layout.size != 0
            let ptr = unsafe { alloc::alloc(layout) } as *mut tensor<MaybeUninit<X>, S>;
            ptr::NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        };
        // SAFETY: see [`Box`] Memory layout section and `Box::try_new_uninit_in`
        unsafe { Box::from_raw(ptr.as_ptr()) }
    }

    /// similar to `&str` and `&[]` literals.
    #[inline]
    pub fn literal<'a>(data: S::Mapped<X>) -> &'a Self {
        Box::leak(Self::new_boxed(data))
    }

    /// Transmutes a reference to the shape into tensor data.
    #[inline]
    pub(crate) fn wrap_ref(data: &S::Mapped<X>) -> &Self {
        unsafe { mem::transmute(data) }
    }

    /// Transmutes a mutable reference to the shape into tensor data.
    #[inline]
    pub(crate) fn wrap_mut(data: &mut S::Mapped<X>) -> &mut Self {
        unsafe { mem::transmute(data) }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn wrap_box(b: Box<S::Mapped<X>>) -> Box<Self> {
        unsafe { mem::transmute(b) }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn unwrap_box(b: Box<Self>) -> Box<S::Mapped<X>> {
        unsafe { mem::transmute(b) }
    }

    #[inline]
    pub fn as_arr(&self) -> &S::Mapped<X> {
        &self.0
    }

    /// Clones `self` into a new [`Box`].
    #[inline]
    pub fn to_box(&self) -> Box<Self> {
        Self::new_boxed(self.0.clone())
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d(&self) -> &vector<X, { S::LEN }> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d_mut<const LEN: usize>(&mut self) -> &mut vector<X, LEN>
    where S: Len<LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    #[inline]
    pub fn transmute_as<S2, const LEN: usize>(&self) -> &tensor<X, S2>
    where
        S: Len<LEN>,
        S2: Shape + Len<LEN>,
    {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    #[inline]
    pub fn transmute_clone<S2, const LEN: usize>(&self) -> Tensor<X, S2>
    where
        S: Len<LEN>,
        S2: Shape + Len<LEN>,
    {
        self.to_owned().transmute_into()
    }

    /// Returns the length of the tensor in its 1D representation.
    #[inline]
    pub fn len(&self) -> usize {
        S::LEN
    }

    #[inline]
    pub fn get_sub_tensor(&self, idx: usize) -> Option<&tensor<X, S::Sub>> {
        self.0.as_sub_slice().get(idx).map(tensor::wrap_ref)
    }

    #[inline]
    pub fn get_sub_tensor_mut(&mut self, idx: usize) -> Option<&mut tensor<X, S::Sub>> {
        self.0.as_mut_sub_slice().get_mut(idx).map(tensor::wrap_mut)
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the
    /// tensor.
    #[inline]
    pub fn iter_sub_tensors<'a>(
        &'a self,
        /*
        ) -> Map<
            slice::Iter<'a, <S::Sub as MultidimArr>::Mapped<X>>,
            impl FnMut(&'a <S::Sub as MultidimArr>::Mapped<X>) -> &'a tensor<X, S::Sub>,
        > {
            */
    ) -> impl Iterator<Item = &'a tensor<X, S::Sub>> {
        self.0.as_sub_slice().iter().map(tensor::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors
    /// of the tensor.
    #[inline]
    pub fn iter_sub_tensors_mut<'a>(
        &'a mut self,
    ) -> Map<
        slice::IterMut<'a, <S::Sub as MultidimArr>::Mapped<X>>,
        impl FnMut(&'a mut <S::Sub as MultidimArr>::Mapped<X>) -> &'a mut tensor<X, S::Sub>,
    > {
        self.0.as_mut_sub_slice().iter_mut().map(tensor::wrap_mut)
    }

    /// Sets the tensor to `val`.
    #[inline]
    pub fn set(&mut self, val: S::Mapped<X>) {
        self.0 = val;
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_clone<Y: Element>(&self, mut f: impl FnMut(X) -> Y) -> Tensor<Y, S> {
        let mut out = Tensor::<Y, S>::new_uninit();
        for (y, &x) in out.iter_elem_mut().zip(self.iter_elem()) {
            y.write(f(x));
        }
        unsafe { mem::transmute(out) }
    }
}

impl<X: Element, S: Shape> Multidimensional<X> for tensor<X, S> {
    type Iter<'a> = slice::Iter<'a, X>;
    type IterMut<'a> = slice::IterMut<'a, X>;

    #[inline]
    fn iter_elem(&self) -> slice::Iter<'_, X> {
        let ptr = self.0.as_ptr();
        unsafe { slice::from_raw_parts(ptr, S::LEN) }.iter()
    }

    #[inline]
    fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X> {
        let ptr = self.0.as_mut_ptr();
        unsafe { slice::from_raw_parts_mut(ptr, S::LEN) }.iter_mut()
    }
}

impl<X: Element, S: Shape> Default for tensor<X, S> {
    fn default() -> Self {
        Self::new(S::Mapped::default())
    }
}

impl<X: Element, S: Shape> PartialEq<Self> for tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<X: Element, S: Shape> Eq for tensor<X, S> where S::Mapped<X>: Eq {}

impl<X: Element, S: Shape, D: MultidimArr<Element = X, Mapped<()> = S>> PartialEq<D>
    for tensor<X, S>
where S::Mapped<X>: PartialEq
{
    fn eq(&self, other: &D) -> bool {
        self.0 == other.type_hint()
    }
}

impl<X: Element, S: Shape> ToOwned for tensor<X, S> {
    type Owned = Tensor<X, S>;

    fn to_owned(&self) -> Self::Owned {
        Tensor::from(self.to_box())
    }
}

impl<X: Element, S: Shape> Index<usize> for tensor<X, S> {
    type Output = tensor<X, S::Sub>;

    fn index(&self, idx: usize) -> &Self::Output {
        self.get_sub_tensor(idx).expect("`idx` is in range of the outermost dimension")
    }
}

impl<X: Element, S: Shape> IndexMut<usize> for tensor<X, S> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.get_sub_tensor_mut(idx)
            .expect("`idx` is in range of the outermost dimension")
    }
}

impl<X: Element + Serialize, S: Shape> Serialize for tensor<X, S> {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where Ser: serde::Serializer {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self.iter_elem() {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

impl<X: Element> scalar<X> {
    /// Returns the value of the [`scalar`].
    #[inline]
    pub fn val(&self) -> X {
        self.0
    }
}
