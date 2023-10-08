use std::ops::{Add, Mul};

pub trait Ring: Sized + Add<Self, Output = Self> + Mul<Self, Output = Self> {
    const ZERO: Self;
    const ONE: Self;
}

macro_rules! impl_ring {
    ( $( $type:ty )+ : $zero:literal $one:literal ) => { $(
        impl Ring for $type {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
        }
    )+ };
}
impl_ring! { i8 i16 i32 i64 i128: 0 1 }
impl_ring! { u8 u16 u32 u64 u128: 0 1 }
impl_ring! { f32 f64: 0.0 1.0 }
