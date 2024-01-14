use crate::{
    data, maybe_uninit::MaybeUninit, shape_data::ArrDefault, tensor, Element, Float, Len, Num,
    Shape, TensorData,
};
use std::{
    alloc,
    borrow::{Borrow, BorrowMut},
    fmt, mem,
    ops::{Deref, DerefMut},
    ptr,
};

#[repr(transparent)]
pub struct Tensor<X: Element, S: Shape> {
    /// [`Box`] pointer to the tensor data.
    data: Box<data::tensor<X, S>>,
}

pub mod aliases {
    use super::Tensor;

    macro_rules! make_aliases {
        ($($name:ident : $($dim_name:ident)* => $shape:ty),* $(,)?) => {
            make_aliases! { inner $( $name : $( $dim_name )* => $shape; stringify!($shape) ),* }
        };
        (inner $($name:ident : $($dim_name:ident)* => $shape:ty; $shape_str:expr),*) => { $(
            /// owned tensor
            #[doc = "()"]
            /// test
            #[allow(non_camel_case_types)]
            pub type $name<X, $(const $dim_name: usize),*> = Tensor<X, $shape>;
        )* };
    }

    make_aliases! {
        Scalar: => (),
        Vector: N => [(); N],
        Matrix: W H => [[(); W]; H],
        Tensor3: A B C => [[[(); A]; B]; C],
        Tensor4: A B C D => [[[[(); A]; B]; C]; D],
        Tensor5: A B C D E => [[[[[(); A]; B]; C]; D]; E],
        Tensor6: A B C D E F => [[[[[[(); A]; B]; C]; D]; E]; F],
        Tensor7: A B C D E F G => [[[[[[[(); A]; B]; C]; D]; E]; F]; G],
        Tensor8: A B C D E F G H => [[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H],
        Tensor9: A B C D E F G H I => [[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I],
        Tensor10: A B C D E F G H I J => [[[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I]; J],
    }
}
use aliases::*;

impl<X: Element, S: Shape> Tensor<X, S> {
    /// Creates a new [`Tensor`] on the heap.
    #[inline]
    pub fn new(data: S::Data<X>) -> Self {
        Self::from(data::tensor::new_boxed(data))
    }

    /// Allocates a new uninitialized [`Tensor`].
    #[inline]
    pub fn new_uninit() -> Tensor<MaybeUninit<X>, S> {
        let ptr = if mem::size_of::<X>() == 0 {
            ptr::NonNull::dangling()
        } else {
            let layout = alloc::Layout::new::<data::tensor<MaybeUninit<X>, S>>();
            // SAFETY: layout.size != 0
            let ptr = unsafe { alloc::alloc(layout) } as *mut data::tensor<MaybeUninit<X>, S>;
            ptr::NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        };
        // SAFETY: see [`Box`] Memory layout section and `Box::try_new_uninit_in`
        Tensor::from(unsafe { Box::from_raw(ptr.as_ptr()) })
    }

    /// Creates a new Tensor filled with the values in `iter`.
    /// If the [`Iterator`] it too small, the rest of the elements contain the value `X::default()`.
    #[inline]
    pub fn from_iter<const LEN: usize>(iter: impl IntoIterator<Item = X>) -> Self
    where S: Len<LEN> {
        let mut t = Self::default();
        t.iter_elem_mut().zip(iter).for_each(|(x, val)| *x = val);
        t
    }

    /// Creates a new Tensor filled with the scalar value.
    #[inline]
    pub fn full<const LEN: usize>(val: X) -> Self
    where S: Len<LEN> {
        Self::from_1d(Vector::new([val; LEN]))
    }

    /// Creates a new Tensor filled with the scalar value `0`.
    #[inline]
    pub fn zeros<const LEN: usize>() -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        Self::full(X::ZERO)
    }

    /// Creates a new Tensor filled with the scalar value `1`.
    #[inline]
    pub fn ones<const LEN: usize>() -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        Self::full(X::ONE)
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    pub fn from_1d<const LEN: usize>(vec: Vector<X, LEN>) -> Self
    where S: Len<LEN> {
        let vec = mem::ManuallyDrop::new(vec);
        unsafe { mem::transmute_copy(&vec) }
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    pub fn into_1d<const LEN: usize>(self) -> Vector<X, LEN>
    where S: Len<LEN> {
        let t = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&t) }
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

    /// Applies a function to every element of the tensor.
    // TODO: bench vs tensor::map_clone
    #[inline]
    pub fn map_inplace<const LEN: usize>(mut self, mut f: impl FnMut(X) -> X) -> Self
    where S: Len<LEN> {
        self.map_mut(|x| *x = f(*x));
        self
    }

    /// Multiplies the tensor by a scalar value.
    #[inline]
    pub fn scalar_mul<const LEN: usize>(mut self, scalar: X) -> Self
    where
        S: Len<LEN>,
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
    pub fn add_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.add_elem_mut(other);
        self
    }

    /// Subtracts `other` from `self` elementwise.
    #[inline]
    pub fn sub_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.sub_elem_mut(other);
        self
    }

    /// Multiplies `other` to `self` elementwise.
    #[inline]
    pub fn mul_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.mul_elem_mut(other);
        self
    }

    /// Calculates the reciprocal of every element in `self`.
    #[inline]
    pub fn recip_elem<const LEN: usize>(mut self) -> Self
    where
        S: Len<LEN>,
        X: Float,
    {
        self.recip_elem_mut();
        self
    }

    /// Calculates the negative of the tensor.
    #[inline]
    pub fn neg<const LEN: usize>(mut self) -> Self
    where
        S: Len<LEN>,
        X: Float,
    {
        self.neg_mut();
        self
    }
}

impl<X: Element + fmt::Debug, S: Shape + fmt::Debug> fmt::Debug for Tensor<X, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tensor").field(&self.0).finish()
    }
}

impl<X: Element, S: Shape> Clone for Tensor<X, S> {
    fn clone(&self) -> Self {
        self.data.to_owned()
    }
}

impl<X: Element, S: Shape> Default for Tensor<X, S> {
    fn default() -> Self {
        Self::new(ArrDefault::arr_default())
    }
}

impl<X: Element + PartialEq, S: Shape> PartialEq for Tensor<X, S>
where S: Len<{ S::LEN }>
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<X: Element + PartialEq, S: Shape> PartialEq<&tensor<X, S>> for Tensor<X, S>
where S: Len<{ S::LEN }>
{
    fn eq(&self, other: &&tensor<X, S>) -> bool {
        self.data.as_ref() == *other
    }
}

impl<X: Element + PartialEq, S: Shape> PartialEq<Tensor<X, S>> for &tensor<X, S>
where S: Len<{ S::LEN }>
{
    fn eq(&self, other: &Tensor<X, S>) -> bool {
        other.data.as_ref() == *self
    }
}

impl<X: Element + Eq, S: Shape> Eq for Tensor<X, S> where S: Len<{ S::LEN }> {}

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

/*
impl<'de, X: Element, S: Shape> Deserialize<'de> for Tensor<X, S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        todo!()
    }
}
*/

/*
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
        Self::from_box(new_boxed_tensor_data(data))
    }

}
*/

#[cfg(test)]
mod tests {
    use super::*;

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
