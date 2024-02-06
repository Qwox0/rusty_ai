use crate::{
    maybe_uninit::MaybeUninit,
    multidim_arr::{Len, MultidimArr},
    multidimensional::{Multidimensional, MultidimensionalOwned},
    owned::Tensor,
    scalar, vector, Element, Shape,
};
use core::mem;
use serde::Serialize;
use std::{
    alloc,
    iter::Map,
    ops::{Index, IndexMut},
    ptr, slice,
};

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

    #[inline]
    pub(crate) fn new_uninit() -> tensor<MaybeUninit<X>, S> {
        // SAFETY: tensor contains MaybeUninit which doesn't need initialization
        unsafe { MaybeUninit::uninit().assume_init() }
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

    /*
    pub fn new_boxed(data: impl ShapeData<Element = X, Shape = S>) -> Box<Self> {
        Box::new(Self::new(data.type_hint()))
    }
    */

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

    #[inline]
    pub(crate) fn wrap_box(b: Box<S::Mapped<X>>) -> Box<Self> {
        unsafe { mem::transmute(b) }
    }

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
        Self::new(S::Mapped::<X>::unwrap(<S::Mapped<X> as MultidimArr>::Wrapped::default()))
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

impl<X: Element + Serialize, S: Shape> Serialize for tensor<X, S>
where <S::Mapped<X> as MultidimArr>::Wrapped: Serialize
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where Ser: serde::Serializer {
        S::Mapped::wrap_ref(&self.0).serialize(serializer)
    }
}

impl<X: Element> scalar<X> {
    /// Returns the value of the [`scalar`].
    #[inline]
    pub fn val(&self) -> X {
        self.0
    }
}

// ===================== old =========================

/*
   pub unsafe trait AsArr<Elem>: Sized {
   type Arr: AsRef<[Elem]> + AsMut<[Elem]> + IndexMut<usize>;

   fn as_arr(&self) -> &Self::Arr {
   unsafe { mem::transmute(self) }
   }

   fn as_arr_mut(&mut self) -> &mut Self::Arr {
   unsafe { mem::transmute(self) }
   }
   }

   unsafe impl<T: Element> AsArr<T> for T {
   type Arr = [T; 1];
   }

   unsafe impl<T, const N: usize> AsArr<T> for [T; N] {
   type Arr = [T; N];
   }

/// Length of a tensor ([`TensorData`]).
///
/// # SAFETY
///
/// Some provided methods on [`TensorData`] requires the correct implementation of this trait.
/// Otherwise undefined behavior might occur.
pub unsafe trait Len<const LEN: usize> {}

pub(crate) fn new_boxed_tensor_data<X: Element, T: TensorData<X>>(data: T::Shape) -> Box<T> {
let data = ManuallyDrop::new(data);
// SAFETY: see TensorData
Box::new(unsafe { mem::transmute_copy::<T::Shape, T>(&data) })
}

/// Trait for Tensor data. Types implementing this trait are similar to [`str`] and `slice` and
/// should only be accessed behind a reference.
///
/// # SAFETY
///
/// * The data structure must be equivalent to [`Box<Self::Data>`] as [`mem::transmute`] and
/// [`mem::transmute_copy`] are used to convert between [`Tensor`] types.
/// * The `LEN` constant has to equal the length of the tensor in its 1D representation.
pub unsafe trait TensorData<X: Element>: Sized + IndexMut<usize> {
/// [`Tensor`] type owning `Self`.
type Owned: Tensor<X, Data = Self>;
/// Internal Shape of the tensor data. Usually an array `[[[[X; A]; B]; ...]; Z]` with the same
/// dimensions as the tensor.
type Shape: Copy + AsArr<<Self::SubData as TensorData<X>>::Shape>;
/// The [`TensorData`] one dimension lower.
type SubData: TensorData<X>;

/// `Self` but with another [`Element`] type.
type Mapped<E: Element>: TensorData<E, Owned = <Self::Owned as Tensor<X>>::Mapped<E>>;

/// The dimension of the tensor.
const DIM: usize;
/// The length of the tensor in its 1D representation.
const LEN: usize;
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    /*
    #[test]
    fn serde_test() {
        let a: &matrix<i32, 2, 2> = tensor::literal([[1, 2], [3, 4]]);
        println!("{:?}", a);
        let json = serde_json::to_string(a).unwrap();
        println!("{:?}", json);
        let a: Matrix<i32, 2, 2> = serde_json::from_str(&json).unwrap();
        println!("{:?}", a);
        panic!()
    }
    */
}
