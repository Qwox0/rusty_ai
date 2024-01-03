use crate::Element;
use core::slice;
use std::mem;

mod private {
    pub trait Sealed {}
}

pub trait Len<const LEN: usize>: private::Sealed {}

impl<S: Shape> private::Sealed for S {}
impl<S: Shape> Len<{ S::LEN }> for S {}
//impl<X: Element, const LEN: usize> Len<LEN> for [X; LEN] {}

pub trait Shape: private::Sealed + Clone + Copy + Sized {
    type Element: Element;
    type SubShape: Shape;

    /// dimension of the tensor data.
    const DIM: usize;
    /// total count of elements in the tensor data.
    const LEN: usize;
    //const SUB_COUNT: usize;

    fn iter_sub<'a>(&'a self) -> slice::Iter<'a, Self::SubShape>;
    fn iter_sub_mut<'a>(&'a mut self) -> slice::IterMut<'a, Self::SubShape>;
}

impl<X: Element> Shape for X {
    type Element = X;
    type SubShape = Self;

    const DIM: usize = 0;
    const LEN: usize = 1;

    //const SUB_COUNT: usize = 1;

    fn iter_sub<'a>(&'a self) -> slice::Iter<'a, Self::SubShape> {
        unsafe { mem::transmute::<&Self, &[X; 1]>(self) }.iter()
    }

    fn iter_sub_mut<'a>(&'a mut self) -> slice::IterMut<'a, Self::SubShape> {
        unsafe { mem::transmute::<&mut Self, &mut [X; 1]>(self) }.iter_mut()
    }
}

impl<X: Element, S: Shape<Element = X>, const N: usize> Shape for [S; N] {
    type Element = X;
    type SubShape = S;

    const DIM: usize = S::DIM + 1;
    const LEN: usize = S::LEN * N;

    //const SUB_COUNT: usize = N;

    fn iter_sub<'a>(&'a self) -> slice::Iter<'a, Self::SubShape> {
        self.iter()
    }

    fn iter_sub_mut<'a>(&'a mut self) -> slice::IterMut<'a, Self::SubShape> {
        self.iter_mut()
    }
}
