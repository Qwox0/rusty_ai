use core::fmt;
use serde::{Deserialize, Serialize};
use std::mem;

/// This wrapper implements [`Default`], [`Serialize`] and [`Deserialize`] for any length `N`. This
/// means that implementations for `Arr<T, 0>` might be overly restrictive.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct Arr<T, const N: usize> {
    #[serde(with = "serde_arrays")]
    #[serde(bound(serialize = "T: Serialize"))]
    #[serde(bound(deserialize = "T: Deserialize<'de>"))]
    pub arr: [T; N],
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Arr<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.arr.fmt(f)
    }
}

impl<T: Default + Copy, const N: usize> Default for Arr<T, N> {
    fn default() -> Self {
        Self { arr: [T::default(); N] }
    }
}

/*
impl<X, SUB, const N: usize> MultidimArr for Arr<SUB, N>
where
    X: Element,
    SUB: MultidimArr<Element = X>,
{
    type Element = X;
    type Mapped<Y: Element> = Arr<SUB::Mapped<Y>, N>;
    type Sub = SUB;

    //type Wrapped = Self;

    const DIM: usize = SUB::DIM + 1;
    const LEN: usize = SUB::LEN * N;

    #[inline]
    fn as_sub_slice(&self) -> &[Self::Sub] {
        self.arr.as_slice()
    }

    #[inline]
    fn as_mut_sub_slice(&mut self) -> &mut [Self::Sub] {
        self.arr.as_mut_slice()
    }
}

pub trait WrapArr {
    type Element;
    type Wrapped: MultidimArr<Element = Self::Element>;

    /// # Example
    ///
    /// ```rust
    /// # use const_tensor::multidim_arr::*;
    /// let arr = [[1, 2], [3, 4]];
    /// let wrap = Arr { arr: [Arr { arr: [1, 2] }, Arr { arr: [3, 4] }] };
    /// assert_eq!(arr.wrap(), wrap);
    /// ```
    fn wrap(self) -> Self::Wrapped;

    fn wrap_ref(&self) -> &Self::Wrapped;

    /// # Example
    ///
    /// ```rust
    /// # use const_tensor::multidim_arr::*;
    /// let arr = [[1, 2], [3, 4]];
    /// let wrap = Arr { arr: [Arr { arr: [1, 2] }, Arr { arr: [3, 4] }] };
    /// assert_eq!(arr, <[[i32; 2]; 2]>::unwrap(wrap));
    /// ```
    fn unwrap(data: Self::Wrapped) -> Self;
}

impl<X: Element, T: MultidimArr<Element = X> + WrapArr<Element = X>, const N: usize> WrapArr
    for [T; N]
{
    type Element = X;
    type Wrapped = Arr<T, N>;

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

    #[inline]
    fn unwrap(data: Self::Wrapped) -> Self {
        let data = mem::ManuallyDrop::new(data);
        // SAFETY: `Arr<T, N> == [T; N]`
        unsafe { mem::transmute_copy(&data) }
    }
}
*/
