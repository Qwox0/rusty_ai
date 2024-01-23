use crate::Element;
use core::fmt;
use std::mem;

pub trait MultidimArr: Sized {
    /// Element of the multidimensional array.
    type Element: Element;

    /// Self with `Element = Y`
    type Mapped<Y: Element>: MultidimArr<Element = Y, Mapped<Self::Element> = Self>;

    /// Next smaller shape
    type Sub: MultidimArr<Element = Self::Element>;

    //type Wrapped: MultidimArr<Element = Self::Element>;

    const DIM: usize;
    const LEN: usize;

    fn as_sub_slice(&self) -> &[Self::Sub];
    fn as_mut_sub_slice(&mut self) -> &mut [Self::Sub];

    /*
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
    */

    /// hint that `impl MultidimArr<Element = X, Mapped<()> = S>` is the same type as
    /// `S::Mapped<X>`.
    #[inline]
    fn type_hint(self) -> <Self::Mapped<()> as MultidimArr>::Mapped<Self::Element> {
        self
    }
}

impl<X: Element> MultidimArr for X {
    type Element = X;
    type Mapped<Y: Element> = Y;
    type Sub = X;

    //type Wrapped = X;

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
}

impl<X, SUB, const N: usize> MultidimArr for [SUB; N]
where
    X: Element,
    SUB: MultidimArr<Element = X>,
{
    type Element = X;
    type Mapped<Y: Element> = [SUB::Mapped<Y>; N];
    type Sub = SUB;

    //type Wrapped = Arr<SUB::Wrapped, N>;

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
}

/// # SAFETY
///
/// `Self::LEN == LEN`!
pub unsafe trait Len<const LEN: usize>: MultidimArr {}

unsafe impl<S: MultidimArr> Len<{ S::LEN }> for S {}
