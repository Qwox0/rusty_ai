/*
use crate::{arr_wrapper::Arr, shape_data::ShapeData, Element};
use core::fmt;
use std::mem;

/// The shape/dimensions of a tensor.
pub unsafe trait Shape: Copy + fmt::Debug + Send + Sync + 'static {
    /// multidimensional array with this shape and element `X`
    type Data<X: Element>: ShapeData<Element = X, Shape = Self, Sub = <Self::SubShape as Shape>::Data<X>>;

    /// Next smaller shape
    type SubShape: Shape;

    /// Same as `Self::Data` but using [`Arr`] instead of arrays.
    ///
    /// # SAFETY
    ///
    /// memory layout has to equal the layout of `Self::Data<X>`.
    /// `mem::transmute::<Self::Data<X>, Self::WrappedData<X>>(data)` has to be valid.
    type WrappedData<X: Element>: Copy + Default;

    /// The dimension of the tensor.
    const DIM: usize;

    /// The total number of elements of a tensor having this shape.
    const LEN: usize;

    fn wrap_data<X: Element>(data: Self::Data<X>) -> Self::WrappedData<X> {
        let data = mem::ManuallyDrop::new(data);
        unsafe { mem::transmute_copy(&data) }
    }

    fn wrap_ref_data<X: Element>(data: &Self::Data<X>) -> &Self::WrappedData<X> {
        unsafe { mem::transmute(data) }
    }

    fn unwrap_data<X: Element>(data: Self::WrappedData<X>) -> Self::Data<X> {
        let data = mem::ManuallyDrop::new(data);
        unsafe { mem::transmute_copy(&data) }
    }

    /// Returns the dimensions of the shape as an array.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use const_tensor::Shape;
    /// type MyShape = [[[[[(); 2]; 5]; 1]; 3]; 9];
    /// let dims = MyShape::get_dims_arr();
    /// assert_eq!(dims, [2, 5, 1, 3, 9]);
    /// ```
    fn get_dims_arr() -> [usize; Self::DIM];

    /// Helper for `get_dims_arr`.
    fn _set_dims_arr<const D: usize>(dims: &mut [usize; D]);
}

unsafe impl Shape for () {
    type Data<X: Element> = X;
    type SubShape = ();
    type WrappedData<X: Element> = Arr<X, 1>;

    const DIM: usize = 0;
    const LEN: usize = 1;

    #[inline]
    fn get_dims_arr() -> [usize; 0] {
        []
    }

    #[inline]
    fn _set_dims_arr<const D: usize>(_dims: &mut [usize; D]) {}
}

unsafe impl<SUB: Shape, const N: usize> Shape for [SUB; N] {
    type Data<X: Element> = [SUB::Data<X>; N];
    type SubShape = SUB;
    type WrappedData<X: Element> = Arr<SUB::WrappedData<X>, N>;

    const DIM: usize = SUB::DIM + 1;
    const LEN: usize = SUB::LEN * N;

    #[inline]
    fn get_dims_arr() -> [usize; Self::DIM] {
        let mut dims = [0; Self::DIM];
        Self::_set_dims_arr(&mut dims);
        dims
    }

    #[inline]
    fn _set_dims_arr<const D: usize>(dims: &mut [usize; D]) {
        dims[Self::DIM - 1] = N;
        SUB::_set_dims_arr(dims);
    }
}

/// Length of a [`Shape`].
// NOTE: don't add the `Self: Shape` bound, as it breaks everything!
// The compiler cannot infer `<[(); N] as Shape>::Data<X> == [X; N]`.
pub unsafe trait Len<const LEN: usize> {}

unsafe impl<S: Shape> Len<{ S::LEN }> for S {}
*/
