use crate::{vector, Element, Float, Num, Tensor};
use core::{mem, slice};
use std::iter::Map;

pub unsafe trait AsArr<Elem>: Sized {
    type Arr: AsRef<[Elem]> + AsMut<[Elem]>;

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

/// Trait for Tensor data. Types implementing this trait are similar to [`str`] and `slice` and
/// should only be accessed behind a reference.
///
/// # SAFETY
///
/// * The data structure must be equivalent to [`Box<Self::Data>`] as [`mem::transmute`] and
/// [`mem::transmute_copy`] are used to convert between [`Tensor`] types.
/// * The `LEN` constant has to equal the length of the tensor in its 1D representation.
pub unsafe trait TensorData<X: Element>: Sized {
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

    /// Creates a new tensor data in a [`Box`].
    fn new_boxed(data: Self::Shape) -> Box<Self>;

    /// Transmutes `self` into the inner value.
    #[inline]
    fn _as_inner(&self) -> &Self::Shape {
        unsafe { mem::transmute(self) }
    }

    /// Transmutes `self` into the inner value.
    #[inline]
    fn _as_inner_mut(&mut self) -> &mut Self::Shape {
        unsafe { mem::transmute(self) }
    }

    /// similar to `&str` and `&[]` literals.
    #[inline]
    fn literal<'a>(data: Self::Shape) -> &'a Self {
        Box::leak(Self::new_boxed(data))
    }

    /// Transmutes a reference to the shape into tensor data.
    #[inline]
    fn wrap_ref(data: &Self::Shape) -> &Self {
        unsafe { mem::transmute(data) }
    }

    /// Transmutes a mutable reference to the shape into tensor data.
    #[inline]
    fn wrap_ref_mut(data: &mut Self::Shape) -> &mut Self {
        unsafe { mem::transmute(data) }
    }

    /// Clones `self` into a new [`Box`].
    #[inline]
    fn to_box(&self) -> Box<Self> {
        //Box::new(tensor { data: self.data.clone() });
        Self::new_boxed(self._as_inner().clone())
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d<const LEN: usize>(&self) -> &vector<X, LEN>
    where Self: Len<LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d_mut<const LEN: usize>(&mut self) -> &mut vector<X, LEN>
    where Self: Len<LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Sets the tensor to `val`.
    #[inline]
    fn set(&mut self, val: Self::Shape) {
        *self._as_inner_mut() = val;
    }

    /// Sets every element of the tensor to the scalar value `val`.
    #[inline]
    fn fill<const LEN: usize>(&mut self, val: X)
    where Self: Len<LEN> {
        self.iter_elem_mut().for_each(|x| *x = val);
    }

    /// Sets every element of the tensor to the scalar value `0`.
    #[inline]
    fn fill_zero<const LEN: usize>(&mut self)
    where
        Self: Len<LEN>,
        X: Num,
    {
        self.fill(X::ZERO)
    }

    /// Sets every element of the tensor to the scalar value `1`.
    #[inline]
    fn fill_one<const LEN: usize>(&mut self)
    where
        Self: Len<LEN>,
        X: Num,
    {
        self.fill(X::ONE)
    }

    /// Changes the Shape of the Tensor.
    #[inline]
    fn transmute_as<U, const LEN: usize>(&self) -> &U
    where
        Self: Len<LEN>,
        U: TensorData<X> + Len<LEN>,
    {
        unsafe { mem::transmute(self) }
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
    fn iter_elem<const LEN: usize>(&self) -> slice::Iter<'_, X>
    where Self: Len<LEN> {
        self.as_1d()._as_inner().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    #[inline]
    fn iter_elem_mut<const LEN: usize>(&mut self) -> slice::IterMut<'_, X>
    where Self: Len<LEN> {
        self.as_1d_mut()._as_inner_mut().iter_mut()
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    #[inline]
    fn iter_sub_tensors<'a>(
        &'a self,
    ) -> Map<
        slice::Iter<'a, <Self::SubData as TensorData<X>>::Shape>,
        impl Fn(&'a <Self::SubData as TensorData<X>>::Shape) -> &'a Self::SubData,
    > {
        self._as_inner().as_arr().as_ref().iter().map(Self::SubData::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    #[inline]
    fn iter_sub_tensors_mut<'a>(
        &'a mut self,
    ) -> Map<
        slice::IterMut<'_, <Self::SubData as TensorData<X>>::Shape>,
        impl Fn(&'a mut <Self::SubData as TensorData<X>>::Shape) -> &'a mut Self::SubData,
    > {
        self._as_inner_mut()
            .as_arr_mut()
            .as_mut()
            .iter_mut()
            .map(Self::SubData::wrap_ref_mut)
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem_mut<const LEN: usize>(&mut self, f: impl FnMut(&mut X))
    where Self: Len<LEN> {
        self.iter_elem_mut().for_each(f);
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_clone<Y: Element, const LEN: usize>(
        &self,
        mut f: impl FnMut(X) -> Y,
    ) -> <Self::Mapped<Y> as TensorData<Y>>::Owned
    where
        Self: Len<LEN>,
        Self::Mapped<Y>: Len<LEN>,
    {
        let mut out: <Self::Mapped<Y> as TensorData<Y>>::Owned = Default::default();
        for (o, &x) in out.iter_elem_mut().zip(self.iter_elem()) {
            *o = f(x);
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
    fn scalar_mul_mut<const LEN: usize>(&mut self, scalar: X)
    where
        Self: Len<LEN>,
        X: Num,
    {
        for x in TensorData::<X>::iter_elem_mut(self) {
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
    fn add_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        Self: Len<LEN>,
        X: Num,
    {
        for (x, y) in TensorData::<X>::iter_elem_mut(self).zip(other.iter_elem()) {
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
    fn sub_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        Self: Len<LEN>,
        X: Num,
    {
        for (x, y) in TensorData::<X>::iter_elem_mut(self).zip(other.iter_elem()) {
            *x -= *y;
        }
    }

    /// Multiplies `other` to `self` elementwise and inplace.
    #[inline]
    fn mul_elem_mut<const LEN: usize>(&mut self, other: &Self)
    where
        Self: Len<LEN>,
        X: Num,
    {
        for (x, y) in TensorData::<X>::iter_elem_mut(self).zip(other.iter_elem()) {
            *x *= *y;
        }
    }

    /// Calculates the reciprocal of every element in `self` inplace.
    #[inline]
    fn recip_elem_mut<const LEN: usize>(&mut self)
    where
        Self: Len<LEN>,
        X: Float,
    {
        for x in TensorData::<X>::iter_elem_mut(self) {
            *x = x.recip();
        }
    }

    /// Calculates the negative of the tensor inplace.
    #[inline]
    fn neg_mut<const LEN: usize>(&mut self)
    where
        Self: Len<LEN>,
        X: Float,
    {
        for x in TensorData::<X>::iter_elem_mut(self) {
            *x = x.neg();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn as_arr() {
        let mat = Matrix::new([[1, 2], [3, 4]]);
        let arr = mat._as_inner().as_arr();
        assert_eq!(arr, &[[1, 2], [3, 4]]);
    }

    #[test]
    fn doc_test() {
        use crate::{Matrix, Tensor, TensorData};
        let mat = Matrix::new([[1, 2], [3, 4]]);
        let mut iter = mat.iter_elem();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }
}
