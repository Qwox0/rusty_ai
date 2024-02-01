use crate::{arr_wrapper::Arr, Element, Shape};
use core::fmt;
use std::mem;

pub trait MultidimArr: Sized + Copy + fmt::Debug + Send + Sync + 'static {
    /// Element of the multidimensional array.
    type Element: Element;

    /// Self with `Element = Y`
    type Mapped<Y: Element>: MultidimArr<
            Element = Y,
            Mapped<Self::Element> = Self,
            Sub = <Self::Sub as MultidimArr>::Mapped<Y>,
        >;

    /// Next smaller shape
    type Sub: MultidimArr<Element = Self::Element>;

    /// Same shape as Self but using the [`Arr`] wrapper.
    type Wrapped: Default + Copy;

    const DIM: usize;
    const LEN: usize;

    fn as_sub_slice(&self) -> &[Self::Sub];
    fn as_mut_sub_slice(&mut self) -> &mut [Self::Sub];

    fn as_ptr(&self) -> *const Self::Element;
    fn as_mut_ptr(&mut self) -> *mut Self::Element;

    /// # Example
    ///
    /// ```rust
    /// # use const_tensor::multidim_arr::*;
    /// let arr = [[1, 2], [3, 4]];
    /// let wrap = Arr { arr: [Arr { arr: [1, 2] }, Arr { arr: [3, 4] }] };
    /// assert_eq!(arr.wrap(), wrap);
    /// ```
    #[inline]
    fn wrap(self) -> Self::Wrapped {
        let arr = mem::ManuallyDrop::new(self);
        // SAFETY: `Arr<T, N> == [T; N]`
        unsafe { mem::transmute_copy(&arr) }
    }

    #[inline]
    fn wrap_ref(&self) -> &Self::Wrapped {
        // SAFETY: `Arr<T, N> == [T; N]`
        unsafe { mem::transmute(self) }
    }

    /// # Example
    ///
    /// ```rust
    /// # use const_tensor::multidim_arr::*;
    /// let arr = [[1, 2], [3, 4]];
    /// let wrap = Arr { arr: [Arr { arr: [1, 2] }, Arr { arr: [3, 4] }] };
    /// assert_eq!(arr, <[[i32; 2]; 2]>::unwrap(wrap));
    /// ```
    #[inline]
    fn unwrap(data: Self::Wrapped) -> Self {
        let data = mem::ManuallyDrop::new(data);
        // SAFETY: `Arr<T, N> == [T; N]`
        unsafe { mem::transmute_copy(&data) }
    }

    /// hint that `impl MultidimArr<Element=Element = X, Mapped<()> = S>` is the same type as
    /// `S::Mapped<X>`.
    #[inline]
    fn type_hint(self) -> <Self::Mapped<()> as MultidimArr>::Mapped<Self::Element> {
        self
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

impl<X: Element> MultidimArr for X {
    type Element = X;
    type Mapped<Y: Element> = Y;
    type Sub = X;
    type Wrapped = X;

    const DIM: usize = 0;
    const LEN: usize = 1;

    #[inline]
    fn as_sub_slice(&self) -> &[X] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&Self, &[X; 1]>(self) }.as_slice()
    }

    #[inline]
    fn as_mut_sub_slice(&mut self) -> &mut [X] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&mut Self, &mut [X; 1]>(self) }.as_mut_slice()
    }

    #[inline]
    fn as_ptr(&self) -> *const X {
        self
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut X {
        self
    }

    #[inline]
    fn get_dims_arr() -> [usize; 0] {
        []
    }

    #[inline]
    fn _set_dims_arr<const D: usize>(_dims: &mut [usize; D]) {}
}

impl<X, SUB, const N: usize> MultidimArr for [SUB; N]
where
    X: Element,
    SUB: MultidimArr<Element = X>,
{
    type Element = SUB::Element;
    type Mapped<Y: Element> = [SUB::Mapped<Y>; N];
    type Sub = SUB;
    type Wrapped = Arr<SUB::Wrapped, N>;

    const DIM: usize = SUB::DIM + 1;
    const LEN: usize = SUB::LEN * N;

    #[inline]
    fn as_sub_slice(&self) -> &[Self::Sub] {
        self.as_slice()
    }

    #[inline]
    fn as_mut_sub_slice(&mut self) -> &mut [Self::Sub] {
        self.as_mut_slice()
    }

    #[inline]
    fn as_ptr(&self) -> *const X {
        self.as_slice().as_ptr() as *const X
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut X {
        self.as_mut_slice().as_mut_ptr() as *mut X
    }

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

/// Length bound check for a [`Shape`].
///
/// # SAFETY
///
/// `Self::LEN == LEN`!
pub unsafe trait Len<const LEN: usize> {}

unsafe impl<S: Shape> Len<{ S::LEN }> for S {}
