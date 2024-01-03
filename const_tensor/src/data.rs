use crate::{Element, Len, Shape};
use core::slice;
use std::{
    iter::{FusedIterator, Map},
    mem,
};

/// similar to [`str`] and `slice`.
///
/// # temporary(?) `LEN` constant
///
/// * total amount of elements in the tensor
/// * length this tensor transmuted to its 1D representation
/// * to remove `unconstrained generic constant` errors
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct tensor<S: Shape, const LEN: usize> {
    data: S,
}

impl<X: Element, S: Shape<Element = X> + Len<LEN>, const LEN: usize> tensor<S, LEN> {
    pub(crate) fn new_boxed(data: S) -> Box<tensor<S, LEN>> {
        Box::new(tensor { data })
    }

    /// TODO
    pub(crate) fn literal<'a>(data: S) -> &'a tensor<S, LEN> {
        Box::leak(Self::new_boxed(data))
    }

    fn wrap_ref(data: &S) -> &tensor<S, LEN> {
        unsafe { mem::transmute(data) }
    }

    fn wrap_ref_mut(data: &mut S) -> &mut tensor<S, LEN> {
        unsafe { mem::transmute(data) }
    }
}

// S::LEN must equal LEN!!
impl<X: Element, S: Shape<Element = X>, const LEN: usize> tensor<S, LEN> {
    /// Clones `self` into a new [`Box`].
    pub fn to_box(&self) -> Box<tensor<S, LEN>> {
        //Box::new(tensor { data: self.data })
        Box::new(tensor { data: self.data.clone() })
    }

    pub(crate) fn into_inner(self) -> S {
        self.data
    }

    pub(crate) fn as_inner(&self) -> &S {
        &self.data
    }

    pub(crate) fn as_inner_mut(&mut self) -> &mut S {
        &mut self.data
    }

    pub fn set(&mut self, val: S) {
        self.data = val;
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d(&self) -> &tensor<[X; LEN], LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d_mut(&mut self) -> &mut tensor<[X; LEN], LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    #[inline]
    pub fn transmute_as<S2: Shape<Element = X> + Len<LEN>>(&self) -> &tensor<S2, LEN> {
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    pub fn iter_elem(&self) -> slice::Iter<'_, X> {
        self.as_1d().as_inner().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    #[inline]
    pub fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X> {
        self.as_1d_mut().as_inner_mut().iter_mut()
    }
}

pub trait TensorData {
    type SubShape;
}

impl<X: Element, S, const LEN: usize> TensorData for tensor<S, LEN>
where S: Shape<Element = X>
{
    type SubShape = S::SubShape;
}

impl<SUB: Shape, const N: usize, const LEN: usize> tensor<[SUB; N], LEN> {
    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    pub fn iter_sub_tensors<'a, const L: usize>(
        &'a self,
        /*) -> impl Iterator<Item = &'a tensor<SUB, { SUB::LEN }>>
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator {*/
    ) -> Map<slice::Iter<'a, SUB>, impl Fn(&'a SUB) -> &'a tensor<SUB, L>>
    where
        SUB: Len<L>,
    {
        self.data.iter_sub().map(tensor::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    pub fn iter_sub_tensors_mut<'a, const L: usize>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut tensor<SUB, L>>
    + DoubleEndedIterator
    + ExactSizeIterator
    + FusedIterator
    where SUB: Len<L> {
        /*) -> Map<
            slice::IterMut<'_, SUB>,
            impl Fn(&mut S::SubShape) -> &mut tensor<S::SubShape, { S::SubShape::LEN }>,
        > {*/
        self.data.iter_sub_mut().map(tensor::wrap_ref_mut)
    }
}
/*
impl<X: Element, S, SUB, const LEN: usize> tensor<S, LEN>
where
    S: Shape<Element = X, SubShape = SUB>,
    SUB: Shape,
{
    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    pub fn iter_sub_tensors<'a, const L: usize>(
        &'a self,
        /*) -> impl Iterator<Item = &'a tensor<SUB, { SUB::LEN }>>
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator {*/
    ) -> Map<slice::Iter<'_, SUB>, impl Fn(&SUB) -> &tensor<SUB, L>>
    where
        SUB: Len<L>,
    {
        self.data.iter_sub().map(tensor::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    pub fn iter_sub_tensors_mut<'a, const L: usize>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut tensor<SUB, L>>
    + DoubleEndedIterator
    + ExactSizeIterator
    + FusedIterator
    where SUB: Len<L> + 'a {
        /*) -> Map<
            slice::IterMut<'_, SUB>,
            impl Fn(&mut S::SubShape) -> &mut tensor<S::SubShape, { S::SubShape::LEN }>,
        > {*/
        self.data.iter_sub_mut().map(tensor::wrap_ref_mut)
    }
}
 */

/// Type alias for a 0D tensor.
pub type ScalarData<X> = tensor<X, 1>;
/// Type alias for a 1D tensor.
pub type VectorData<X, const LEN: usize> = tensor<[X; LEN], LEN>;
/// Type alias for a 2D tensor.
pub type MatrixData<X, const W: usize, const H: usize> = tensor<[[X; W]; H], { W * H }>;
/// Type alias for a 3D tensor.
pub type Tensor3Data<X, const A: usize, const B: usize, const C: usize> =
    tensor<[[[X; A]; B]; C], { A * B * C }>;
/// Type alias for a 4D tensor.
pub type Tensor4Data<X, const A: usize, const B: usize, const C: usize, const D: usize> =
    tensor<[[[[X; A]; B]; C]; D], { A * B * C * D }>;

fn test() {
    let a: Box<tensor<i32, 1>> = tensor::new_boxed(333);
}

/*
impl<X: Element> Deref for tensor< 1> {
    type Target = X;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
*/
