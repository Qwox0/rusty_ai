use crate::{tensor, Element, Len, Shape};
use std::{
    borrow::Borrow,
    mem,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Tensor<S: Shape, const LEN: usize> {
    data: Box<tensor<S, LEN>>,
}

impl<X: Element, S: Shape<Element = X>, const LEN: usize> Tensor<S, LEN> {
    /// Creates a new Tensor.
    #[inline]
    pub fn new(data: S) -> Tensor<S, LEN>
    where S: Len<LEN> {
        Tensor::from_box(tensor::new_boxed(data))
    }
}

impl<X: Element, S: Shape<Element = X>, const LEN: usize> Tensor<S, LEN> {
    /// Creates a new Tensor.
    #[inline]
    pub fn from_box(data: Box<tensor<S, LEN>>) -> Tensor<S, LEN> {
        Tensor { data }
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    pub fn from_1d(vec: Tensor<[X; LEN], LEN>) -> Tensor<S, LEN>
    where [X; LEN]: Len<LEN> {
        unsafe { mem::transmute(vec) }
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    pub fn into_1d(self) -> Tensor<[X; LEN], LEN>
    where [X; LEN]: Len<LEN> {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    pub fn transmute_into<S2: Shape<Element = X> + Len<LEN>>(self) -> Tensor<S2, LEN> {
        //Tensor::from_1d(self.into_1d())
        unsafe { mem::transmute(self) }
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem_mut(mut self, f: impl FnMut(&mut X)) -> Self {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem(self, mut f: impl FnMut(X) -> X) -> Self {
        self.map_elem_mut(|x| *x = f(*x))
    }
}

impl<S: Shape, const LEN: usize> Deref for Tensor<S, LEN> {
    type Target = tensor<S, LEN>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<S: Shape, const LEN: usize> DerefMut for Tensor<S, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<S: Shape, const LEN: usize> Borrow<tensor<S, LEN>> for Tensor<S, LEN> {
    fn borrow(&self) -> &tensor<S, LEN> {
        &self.data
    }
}

impl<S: Shape + Len<LEN>, const LEN: usize> From<Box<tensor<S, LEN>>> for Tensor<S, LEN> {
    #[inline]
    fn from(data: Box<tensor<S, LEN>>) -> Self {
        Self::from_box(data)
    }
}

impl<S: Shape + Len<LEN>, const LEN: usize> AsRef<tensor<S, LEN>> for Tensor<S, LEN> {
    #[inline]
    fn as_ref(&self) -> &tensor<S, LEN> {
        &self.data
    }
}

impl<S: Shape + Len<LEN>, const LEN: usize> AsMut<tensor<S, LEN>> for Tensor<S, LEN> {
    #[inline]
    fn as_mut(&mut self) -> &mut tensor<S, LEN> {
        &mut self.data
    }
}

/// Type alias for a 0D tensor.
pub type Scalar<X> = Tensor<X, 1>;
/// Type alias for a 1D tensor.
pub type Vector<X, const LEN: usize> = Tensor<[X; LEN], LEN>;
/// Type alias for a 2D tensor.
pub type Matrix<X, const W: usize, const H: usize> = Tensor<[[X; W]; H], { W * H }>;
/// Type alias for a 3D tensor.
pub type Tensor3<X, const A: usize, const B: usize, const C: usize> =
    Tensor<[[[X; A]; B]; C], { A * B * C }>;
/// Type alias for a 4D tensor.
pub type Tensor4<X, const A: usize, const B: usize, const C: usize, const D: usize> =
    Tensor<[[[[X; A]; B]; C]; D], { A * B * C * D }>;
