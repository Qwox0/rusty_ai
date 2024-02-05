use core::fmt;
use half::{bf16, f16};
use inline_closure::inline_closure;
use num::Zero;
use rand_distr::uniform::SampleUniform;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    iter::{Product, Sum},
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign},
};

/// Represents a single value in a multidimensional object. Like a 0D Tensor.
pub trait Element:
    Copy + Default + PartialEq + fmt::Debug + Serialize + Send + Sync + 'static
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
    ()
    isize i8 i16 i32 i64 i128
    usize u8 u16 u32 u64 u128
    bool
    f32 f64
    f16 bf16
}

/// An [`Element`] which also implements some numeric operations.
pub trait Num:
    Element
    + PartialOrd
    + num::ToPrimitive
    + num::FromPrimitive
    + num::Num
    + num::NumCast
    + MoreNumOps
    + SampleUniform // Remove this?
{
    /// Additive identity of `Self`
    const ZERO: Self;
    /// Multiplicative identity of `Self`
    const ONE: Self;
    /// Maximum value of `Self`.
    const MAX: Self;
    /// Minimum value of `Self`.
    const MIN: Self;

    /// Creates a new number from an [`i32`]. This should be used for simple values which don't
    /// result in precision problems.
    /* const */
    fn lit(lit: i32) -> Self;

    /// Returns `true` if `self` is positive and `false` if the number is zero or negative.
    ///
    /// For floats `0` counts as positive and `-0` counts as negative.
    fn is_positive(self) -> bool;

    /// Defines the overflow behavior of the `cast` method.
    fn clamp_overflow(is_pos_overflow: bool) -> Self;

    /// Returns `Self::MAX` or `Self::MIN` based on `is_pos`.
    #[inline]
    fn signed_max(is_pos: bool) -> Self {
        if is_pos { Self::MAX } else { Self::MIN }
    }

    /// like `as` cast. Results aren't always the same:
    ///
    /// ```rust
    /// # use const_tensor::Num;
    /// assert_eq!(-1i8 as u8, 255);
    /// assert_eq!((-1i8).cast::<u8>(), 0);
    /// ```
    #[inline]
    fn cast<X: Num>(self) -> X {
        X::from(self).unwrap_or_else(|| X::clamp_overflow(self.is_positive()))
    }

    /// Returns `Self::ZERO` or `Self::ONE` based on the [`bool`].
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
    is_positive: |self| !self.is_zero(),
    clamp_overflow: Self::signed_max,
}

impl_num! {
    isize i8 i16 i32 i64 i128,
    0, 1,
    lit: |lit| lit as Self,
    is_positive: |self| self.is_positive(),
    clamp_overflow: Self::signed_max,
}

/// Some numeric operations required for a type to implement [`Num`].
///
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

/// A [`Num`] which also implements some floating point operations.
///
/// This trait is automatically implemented.
pub trait Float: Num + num::Float {
    /// Creates a new float from a [`f32`]. This should be used for simple values which don't
    /// result in precision problems.
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

#[cfg(test)]
mod tests {
    #[test]
    fn is_zero() {
        assert!(!num::Zero::is_zero(&1i32), "1i32 must not be zero");
        assert!(num::Zero::is_zero(&0i32), "0i32 must be zero");

        assert!(!num::Zero::is_zero(&1u32), "1u32 must not be zero");
        assert!(num::Zero::is_zero(&0u32), "0u32 must be zero");

        assert!(!num::Zero::is_zero(&1f32), "1f32 must not be zero");
        assert!(num::Zero::is_zero(&0f32), "0f32 must be zero");
        assert!(num::Zero::is_zero(&-0f32), "-0f32 must be zero");
        assert!(0.0 == -0.0);
    }
}
