use crate::{
    maybe_uninit::MaybeUninit,
    owned::Tensor,
    shape::{Len, Shape},
    shape_data::{ArrDefault, ShapeData},
    Element, Float, Num,
};
use core::{mem, slice};
use std::{
    iter::Map,
    ops::{Add, Index, IndexMut},
};

/// implements [`TensorData`]
#[derive(Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct tensor<X: Element, S: Shape>(
    /// inner [`TensorData`] value
    pub(crate) S::Data<X>,
);

pub trait TensorData<X: Element, S: Shape> {
    type SubTensor: TensorData<X, S::SubShape>;
}

impl<X: Element, S: Shape> TensorData<X, S> for tensor<X, S> {
    type SubTensor = tensor<X, S::SubShape>;
}

pub mod aliases {
    use super::tensor;

    macro_rules! make_aliases {
        ($($name:ident : $($dim_name:ident)* => $shape:ty),* $(,)?) => { $(
            /// tensor data
            #[allow(non_camel_case_types)]
            pub type $name<X, $(const $dim_name: usize),*> = tensor<X, $shape>;
        )* };
    }

    make_aliases! {
        scalar: => (),
        vector: N => [(); N],
        matrix: W H => [[(); W]; H],
        tensor3: A B C => [[[(); A]; B]; C],
        tensor4: A B C D => [[[[(); A]; B]; C]; D],
        tensor5: A B C D E => [[[[[(); A]; B]; C]; D]; E],
        tensor6: A B C D E F => [[[[[[(); A]; B]; C]; D]; E]; F],
        tensor7: A B C D E F G => [[[[[[[(); A]; B]; C]; D]; E]; F]; G],
        tensor8: A B C D E F G H => [[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H],
        tensor9: A B C D E F G H I => [[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I],
        tensor10: A B C D E F G H I J => [[[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I]; J],
    }
}
use aliases::*;

impl<X: Element, S: Shape> tensor<X, S> {
    /// a tensor must be allocated on the heap -> use `new_boxed` or [`Tensor`].
    #[inline]
    pub(crate) fn new(data: S::Data<X>) -> Self {
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
    pub fn new_boxed(data: S::Data<X>) -> Box<Self> {
        Box::new(Self::new(data))
    }

    /// similar to `&str` and `&[]` literals.
    #[inline]
    pub fn literal<'a>(data: S::Data<X>) -> &'a Self {
        Box::leak(Self::new_boxed(data))
    }

    /// Transmutes a reference to the shape into tensor data.
    #[inline]
    pub(crate) fn wrap_ref(data: &S::Data<X>) -> &Self {
        unsafe { mem::transmute(data) }
    }

    /// Transmutes a mutable reference to the shape into tensor data.
    #[inline]
    pub(crate) fn wrap_mut(data: &mut S::Data<X>) -> &mut Self {
        unsafe { mem::transmute(data) }
    }

    /// Clones `self` into a new [`Box`].
    #[inline]
    pub fn to_box(&self) -> Box<Self> {
        Self::new_boxed(self.0.clone())
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d<const LEN: usize>(&self) -> &vector<X, LEN>
    where S: Len<LEN> {
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

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mat = Matrix::new([[1, 2], [3, 4]]);
    /// let mut iter = mat.iter_elem();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter_elem<const LEN: usize>(&self) -> slice::Iter<'_, X>
    where S: Len<LEN> {
        self.as_1d().0.iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    #[inline]
    pub fn iter_elem_mut<const LEN: usize>(&mut self) -> slice::IterMut<'_, X>
    where S: Len<LEN> {
        self.as_1d_mut().0.iter_mut()
    }

    #[inline]
    pub fn get_sub_tensor(&self, idx: usize) -> Option<&tensor<X, S::SubShape>> {
        self.0.as_slice().get(idx).map(|a| unsafe { mem::transmute(a) })
    }

    #[inline]
    pub fn get_sub_tensor_mut(&mut self, idx: usize) -> Option<&mut tensor<X, S::SubShape>> {
        self.0.as_mut_slice().get_mut(idx).map(|a| unsafe { mem::transmute(a) })
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the
    /// tensor.
    #[inline]
    pub fn iter_sub_tensors<'a>(
        &'a self,
    ) -> Map<
        slice::Iter<'a, <S::SubShape as Shape>::Data<X>>,
        impl FnMut(&'a <S::SubShape as Shape>::Data<X>) -> &'a tensor<X, S::SubShape>,
    > {
        self.0.as_slice().iter().map(tensor::<X, S::SubShape>::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors
    /// of the tensor.
    #[inline]
    pub fn iter_sub_tensors_mut<'a>(
        &'a mut self,
    ) -> Map<
        slice::IterMut<'a, <S::SubShape as Shape>::Data<X>>,
        impl FnMut(&'a mut <S::SubShape as Shape>::Data<X>) -> &'a mut tensor<X, S::SubShape>,
    > {
        self.0.as_mut_slice().iter_mut().map(tensor::<X, S::SubShape>::wrap_mut)
    }

    /// Sets the tensor to `val`.
    #[inline]
    pub fn set(&mut self, val: S::Data<X>) {
        self.0 = val;
    }

    /// Sets every element of the tensor to the scalar value `val`.
    #[inline]
    pub fn fill<const LEN: usize>(&mut self, val: X)
    where S: Len<LEN> {
        self.iter_elem_mut().for_each(|x| *x = val);
    }

    /// Sets every element of the tensor to the scalar value `0`.
    #[inline]
    pub fn fill_zero<const LEN: usize>(&mut self)
    where
        S: Len<LEN>,
        X: Num,
    {
        self.fill(X::ZERO)
    }

    /// Sets every element of the tensor to the scalar value `1`.
    #[inline]
    pub fn fill_one<const LEN: usize>(&mut self)
    where
        S: Len<LEN>,
        X: Num,
    {
        self.fill(X::ONE)
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_mut<const LEN: usize>(&mut self, f: impl FnMut(&mut X))
    where S: Len<LEN> {
        self.iter_elem_mut().for_each(f);
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_clone<Y: Element, const LEN: usize>(
        &self,
        mut f: impl FnMut(X) -> Y,
    ) -> Tensor<Y, S>
    where
        S: Len<LEN>,
    {
        let mut out: Tensor<Y, S> = Default::default(); // TODO: uninit
        for (y, &x) in out.iter_elem_mut().zip(self.iter_elem()) {
            *y = f(x);
        }
        out
    }

    /// Multiplies the tensor by a scalar value inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat = Matrix::new([[1, 2], [3, 4]]);
    /// mat.scalar_mul_mut(10);
    /// assert_eq!(mat._as_inner(), &[[10, 20], [30, 40]]);
    /// ```
    #[inline]
    pub fn scalar_mul_mut<const LEN: usize>(&mut self, scalar: X)
    where
        S: Len<LEN>,
        X: Num,
    {
        for x in self.iter_elem_mut() {
            *x *= scalar;
        }
    }

    /// Adds `other` to `self` elementwise and inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[1, 2], [3, 4]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// mat1.add_elem_mut(&mat2);
    /// assert_eq!(mat1._as_inner(), &[[5, 5], [5, 5]]);
    /// ```
    #[inline]
    pub fn add_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        S: Len<LEN>,
        X: Num,
    {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x += *y;
        }
    }

    /// Subtracts `other` from `self` elementwise and inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[5, 5], [5, 5]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// mat1.sub_elem_mut(&mat2);
    /// assert_eq!(mat1._as_inner(), &[[1, 2], [3, 4]]);
    /// ```
    #[inline]
    pub fn sub_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        S: Len<LEN>,
        X: Num,
    {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x -= *y;
        }
    }

    /// Multiplies `other` to `self` elementwise and inplace.
    #[inline]
    pub fn mul_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        S: Len<LEN>,
        X: Num,
    {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x *= *y;
        }
    }

    /// Calculates the reciprocal of every element in `self` inplace.
    #[inline]
    pub fn recip_elem_mut<const LEN: usize>(&mut self)
    where
        S: Len<LEN>,
        X: Float,
    {
        for x in self.iter_elem_mut() {
            *x = x.recip();
        }
    }

    /// Calculates the negative of the tensor inplace.
    #[inline]
    pub fn neg_mut<const LEN: usize>(&mut self)
    where
        S: Len<LEN>,
        X: Float,
    {
        for x in self.iter_elem_mut() {
            *x = x.neg();
        }
    }
}

impl<X: Element + Default, S: Shape + Default> Default for tensor<X, S> {
    fn default() -> Self {
        Self::new(ArrDefault::arr_default())
    }
}

impl<X: Element + PartialEq, S: Shape> PartialEq for tensor<X, S>
where S: Len<{ S::LEN }>
{
    fn eq(&self, other: &Self) -> bool {
        self.as_1d().0 == other.as_1d().0
    }
}

impl<X: Element + Eq, S: Shape> Eq for tensor<X, S> where S: Len<{ S::LEN }> {}

impl<X: Element, S: Shape> ToOwned for tensor<X, S> {
    type Owned = Tensor<X, S>;

    fn to_owned(&self) -> Self::Owned {
        Tensor::from(self.to_box())
    }
}

impl<X: Element, S: Shape> Index<usize> for tensor<X, S> {
    type Output = tensor<X, S::SubShape>;

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
