use crate::Len;
use core::fmt;
use half::{bf16, f16};
use inline_closure::inline_closure;
use rand_distr::uniform::SampleUniform;
use std::{
    iter::{Product, Sum},
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign},
};

/// Represents a single value in a multidimensional object. Like a 0D Tensor.
///
/// This trait is automatically implemented.
pub trait Element:
    Copy + Len<1> + Default + fmt::Debug + fmt::Display + Send + Sync + 'static
{
}

macro_rules! impl_element {
    ($($ty:ty)+) => {
        $(
            impl Element for $ty { }
        )+
    };
}

impl_element! {
    isize i8 i16 i32 i64 i128
    usize u8 u16 u32 u64 u128
    bool
    f32 f64
    f16 bf16
}

pub trait Num:
    Element
    + num::ToPrimitive
    + num::FromPrimitive
    + num::Num
    + num::NumCast
    + MoreNumOps
    + SampleUniform // Remove this?
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
    const MIN: Self;

    /* const */
    fn lit(lit: i32) -> Self;

    /// Returns `true` if `self` is positive and `false` if the number is zero or negative.
    fn is_positive(self) -> bool;

    fn clamp_overflow(is_pos_overflow: bool) -> Self;

    #[inline]
    fn signed_max(is_pos: bool) -> Self {
        if is_pos { Self::MAX } else { Self::MIN }
    }

    /// like `as` cast. Results aren't always the same:
    ///
    /// ```rust
    /// # use matrix::Num;
    /// assert_eq!(-1i8 as u8, 255);
    /// assert_eq!((-1i8).cast::<u8>(), 0);
    /// ```
    #[inline]
    fn cast<X: Num>(self) -> X {
        X::from(self).unwrap_or_else(|| X::clamp_overflow(self.is_positive()))
    }

    /// Returns `Self::ZERO` or `Self::ONE` based on the `bool`.
    fn from_bool(b: bool) -> Self {
        if b { Self::ONE } else { Self::ZERO }
    }
}

macro_rules! impl_num {
    (
        $($ty:ty)*,
        $zero:expr, $one:expr,
        lit: $lit:expr,
        is_positive: $is_positive:expr,
        clamp_overflow: $clamp_overflow:expr $(,)?
    ) => { $(
        impl Num for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const MAX: Self = <$ty>::MAX;
            const MIN: Self = <$ty>::MIN;

            #[inline]
            fn lit(lit: i32) -> Self {
                inline_closure!($lit)
            }

            #[inline]
            fn is_positive(self) -> bool {
                inline_closure!($is_positive)
            }

            #[inline]
            fn clamp_overflow(is_pos_overflow: bool) -> Self {
                $clamp_overflow(is_pos_overflow)
            }
        }
    )* };
}

impl_num! {
    f16 bf16,
    Self::ZERO, Self::ONE,
    lit: |lit| Self::from_f32(lit as f32),
    is_positive: |self| self.is_sign_positive(),
    clamp_overflow: Self::signed_infinity,
}

impl_num! {
    f32 f64,
    0.0, 1.0,
    lit: |lit| lit as Self,
    is_positive: |self| self.is_sign_positive(),
    clamp_overflow: Self::signed_infinity,
}

impl_num! {
    usize u8 u16 u32 u64 u128,
    0, 1,
    lit: |lit| lit as Self,
    is_positive: |_| true,
    clamp_overflow: Self::signed_max,
}

impl_num! {
    isize i8 i16 i32 i64 i128,
    0, 1,
    lit: |lit| lit as Self,
    is_positive: |self| self.is_positive(),
    clamp_overflow: Self::signed_max,
}

/// This trait is automatically implemented.
pub trait MoreNumOps:
    AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Sum
    + for<'a> Sum<&'a Self>
    + Product
    + for<'a> Product<&'a Self>
{
}

impl<
    T: AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Sum
        + for<'a> Sum<&'a Self>
        + Product
        + for<'a> Product<&'a Self>,
> MoreNumOps for T
{
}

/// This trait is automatically implemented.
pub trait Float: Num + num::Float {
    fn f_lit(lit: f32) -> Self {
        Self::from_f32(lit).unwrap_or_else(|| Self::clamp_overflow(lit.is_positive()))
    }
    /// Returns infinity with the sign based on `is_pos`.
    #[inline]
    fn signed_infinity(is_pos: bool) -> Self {
        if is_pos { Self::infinity() } else { Self::neg_infinity() }
    }
}

impl<T: Num + num::Float> Float for T {}

//impl Float for half::f16 {}
