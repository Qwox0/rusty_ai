use crate::{shape_data::ShapeData, Element};
use core::fmt;

/// The shape/dimensions of a tensor.
pub trait Shape: Copy + fmt::Debug + Send + Sync + 'static {
    /// multidimensional array with this shape and element `X`
    type Data<X: Element>: ShapeData<<Self::SubShape as Shape>::Data<X>>;
    /// Next smaller shape
    type SubShape: Shape;

    /// The dimensions of the tensor.
    const DIM: usize;
    /// The total number of elements of a tensor having this shape.
    const LEN: usize;

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

impl Shape for () {
    type Data<X: Element> = X;
    type SubShape = ();

    const DIM: usize = 0;
    const LEN: usize = 1;

    #[inline]
    fn get_dims_arr() -> [usize; 0] {
        []
    }

    #[inline]
    fn _set_dims_arr<const D: usize>(_dims: &mut [usize; D]) {}
}

impl<SUB: Shape, const N: usize> Shape for [SUB; N] {
    type Data<X: Element> = [SUB::Data<X>; N];
    type SubShape = SUB;

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
