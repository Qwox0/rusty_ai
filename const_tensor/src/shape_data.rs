use crate::{Element, Shape};
use core::fmt;
use std::mem;

pub trait ShapeData: Sized + Copy + PartialEq + fmt::Debug + Send + Sync + 'static {
    type Element: Element;
    type Shape: Shape<Data<Self::Element> = Self>;
    type Sub: ShapeData;

    fn as_slice(&self) -> &[Self::Sub];
    fn as_mut_slice(&mut self) -> &mut [Self::Sub];

    fn type_hint(self) -> <Self::Shape as Shape>::Data<Self::Element> {
        self
    }
}

impl<X: Element> ShapeData for X {
    type Element = X;
    type Shape = ();
    type Sub = X;

    #[inline]
    fn as_slice(&self) -> &[X] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&Self, &[X; 1]>(self) }.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [X] {
        // SAFETY: T == [T; 1]
        unsafe { mem::transmute::<&mut Self, &mut [X; 1]>(self) }.as_mut_slice()
    }
}

impl<X: Element, SUB: ShapeData<Element = X>, const N: usize> ShapeData for [SUB; N] {
    type Element = X;
    type Shape = [SUB::Shape; N];
    type Sub = SUB;

    #[inline]
    fn as_slice(&self) -> &[SUB] {
        self.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [SUB] {
        self.as_mut_slice()
    }
}

/*
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
*/

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
