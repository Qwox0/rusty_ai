use crate::Element;
use core::fmt;
use std::mem;

pub trait ShapeData<SUB>: Sized + Copy + ArrDefault + fmt::Debug + Send + Sync + 'static {
    fn as_slice(&self) -> &[SUB];
    fn as_mut_slice(&mut self) -> &mut [SUB];
}

impl<T: Element> ShapeData<T> for T {
    #[inline]
    fn as_slice(&self) -> &[T] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&Self, &[T; 1]>(self) }.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&mut Self, &mut [T; 1]>(self) }.as_mut_slice()
    }
}

impl<SUB: Copy + ArrDefault + fmt::Debug + Send + Sync + 'static, const N: usize> ShapeData<SUB>
    for [SUB; N]
{
    #[inline]
    fn as_slice(&self) -> &[SUB] {
        self.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [SUB] {
        self.as_mut_slice()
    }
}

pub trait ArrDefault {
    fn arr_default() -> Self;
}

impl<T: ArrDefault + Copy, const N: usize> ArrDefault for [T; N] {
    #[inline]
    fn arr_default() -> Self {
        [T::arr_default(); N]
    }
}

impl<T: Element> ArrDefault for T {
    #[inline]
    fn arr_default() -> Self {
        T::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_slice_safety() {
        let x = 5;
        let slice = x.as_slice();
        assert_eq!(slice[0], x);
        assert!(slice.len() == 1);
    }
}
