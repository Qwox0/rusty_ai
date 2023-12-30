use crate::{private, Element, TensorData, TensorI};
use core::slice;
use std::{
    mem,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Tensor<D: TensorData> {
    data: Box<D>,
}

impl<X: Element, D: TensorData<Element = X>> private::Sealed<X> for Tensor<D> {}
impl<X: Element, D: TensorData<Element = X>> TensorI<X, { D::LEN }> for Tensor<D> {
    type Data = D;

    #[inline]
    fn into_inner(self) -> Box<Self::Data> {
        self.data
    }
}

impl<D: TensorData> Deref for Tensor<D> {
    type Target = D;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<D: TensorData> DerefMut for Tensor<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<X, D> Tensor<D>
where
    X: Element,
    D: TensorData<Element = X>,
{
    /// Creates a new Tensor.
    #[inline]
    pub fn new(data: D) -> Tensor<D> {
        Tensor::from_box(Box::new(data))
    }

    /// Creates a new Tensor.
    #[inline]
    pub fn from_box(data: Box<D>) -> Tensor<D> {
        let data = data.into();
        Tensor { data }
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    pub fn from_1d(vec: Tensor<[X; D::LEN]>) -> Tensor<D> {
        TensorI::from_1d(vec)
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    pub fn into_1d(self) -> Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    pub fn transmute_into<T: TensorI<X, { D::LEN }>>(self) -> T {
        T::from_1d(self.into_1d())
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem_mut(mut self, f: impl FnMut(&mut X)) -> Self
    where [X; D::LEN]: Sized {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem(self, mut f: impl FnMut(X) -> X) -> Self
    where [X; D::LEN]: Sized {
        self.map_elem_mut(|x| *x = f(*x))
    }
}

/*
impl<X, D> Tensor<D>
where
    X: Element,
    D: TensorData<Element = X>,
{
    /// dimension of the tensor.
    pub const DIM: usize = D::DIM;
    /// total count of elements in the tensor.
    pub const LEN: usize = D::LEN;

    /// Creates a new Tensor.
    #[inline]
    pub fn new(data: D) -> Tensor<D> {
        Tensor::from_box(Box::new(data))
    }

    /// Creates a new Tensor.
    #[inline]
    pub fn from_box(data: Box<D>) -> Tensor<D> {
        let data = data.into();
        Tensor { data }
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    pub fn from_1d(vec: Tensor<[X; D::LEN]>) -> Self {
        TensorI::from_1d(vec)
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    pub fn into_1d(self) -> Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    pub fn transmute<T: TensorI<X, { D::LEN }>>(self) -> T {
        T::from_1d(self.into_1d())
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d(&self) -> &Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    pub fn as_1d_mut(&mut self) -> &mut Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    #[inline]
    pub fn iter_sub_tensors<'a>(&'a self) -> slice::Iter<'a, D::SubData> {
        self.as_ref().iter_sub_tensors()
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    #[inline]
    pub fn iter_sub_tensors_mut<'a>(&'a mut self) -> slice::IterMut<'a, D::SubData> {
        self.as_mut().iter_sub_tensors_mut()
    }

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    pub fn iter_elem(&self) -> slice::Iter<'_, X>
    where [X; D::LEN]: Sized {
        self.as_1d().as_ref().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    pub fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X>
    where [X; D::LEN]: Sized {
        self.as_1d_mut().as_mut().iter_mut()
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem_mut(mut self, f: impl FnMut(&mut X)) -> Self
    where [X; D::LEN]: Sized {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    pub fn map_elem(self, mut f: impl FnMut(X) -> X) -> Self
    where [X; D::LEN]: Sized {
        self.map_elem_mut(|x| *x = f(*x))
    }
}
*/

// ===========================================

/*
pub trait TensorOpts<X: Element, D: TensorData<Element = X>> {
    /// dimension of the tensor.
    const DIM: usize;
    /// total count of elements in the tensor.
    const LEN: usize;

    /// Converts the Tensor into the 1D representation of its elements.
    fn into_1d(self) -> Tensor<[X; D::LEN]>;

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    fn transmute<T: TensorI<X, { D::LEN }>>(self) -> T;

    /// Creates a reference to the elements of the tensor in its 1D representation.
    fn as_1d(&self) -> &Tensor<[X; D::LEN]>;

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    fn as_1d_mut(&mut self) -> &mut Tensor<[X; D::LEN]>;

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    fn iter_sub_tensors<'a>(&'a self) -> slice::Iter<'a, D::SubData>;

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    fn iter_sub_tensors_mut<'a>(&'a mut self) -> slice::IterMut<'a, D::SubData>;

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    fn iter_elem(&self) -> slice::Iter<'_, X>
    where [X; D::LEN]: Sized;

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X>
    where [X; D::LEN]: Sized;

    /// Applies a function to every element of the tensor.
    fn map_elem_mut(self, f: impl FnMut(&mut X)) -> Self
    where [X; D::LEN]: Sized;

    /// Applies a function to every element of the tensor.
    fn map_elem(self, f: impl FnMut(X) -> X) -> Self
    where [X; D::LEN]: Sized;
}

impl<X, D> Tensor<D>
where
    X: Element,
    D: TensorData<Element = X>,
{
    /// Creates a new Tensor.
    #[inline]
    pub fn new(data: D) -> Tensor<D> {
        Tensor::from_box(Box::new(data))
    }

    /// Creates a new Tensor.
    #[inline]
    pub fn from_box(data: Box<D>) -> Tensor<D> {
        let data = data.into();
        Tensor { data }
    }
}

impl<X, D> TensorOpts<X, D> for Tensor<D>
where
    X: Element,
    D: TensorData<Element = X>,
{
    /// dimension of the tensor.
    const DIM: usize = D::DIM;
    /// total count of elements in the tensor.
    const LEN: usize = D::LEN;

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    fn into_1d(self) -> Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    fn transmute<T: TensorI<X, { D::LEN }>>(self) -> T {
        T::from_1d(self.into_1d())
    }

    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d(&self) -> &Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d_mut(&mut self) -> &mut Tensor<[X; D::LEN]> {
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    #[inline]
    fn iter_sub_tensors<'a>(&'a self) -> slice::Iter<'a, D::SubData> {
        self.as_ref().iter_sub_tensors()
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    #[inline]
    fn iter_sub_tensors_mut<'a>(&'a mut self) -> slice::IterMut<'a, D::SubData> {
        self.as_mut().iter_sub_tensors_mut()
    }

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    fn iter_elem(&self) -> slice::Iter<'_, X>
    where [X; D::LEN]: Sized {
        self.as_1d().as_ref().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    ///
    /// Alias for `tensor.as_1d().iter()`.
    #[inline]
    fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X>
    where [X; D::LEN]: Sized {
        self.as_1d_mut().as_mut().iter_mut()
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem_mut(mut self, f: impl FnMut(&mut X)) -> Self
    where [X; D::LEN]: Sized {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem(self, mut f: impl FnMut(X) -> X) -> Self
    where [X; D::LEN]: Sized {
        self.map_elem_mut(|x| *x = f(*x))
    }
}
*/

impl<D: TensorData> From<Box<D>> for Tensor<D> {
    #[inline]
    fn from(data: Box<D>) -> Self {
        Self::from_box(data)
    }
}

impl<D: TensorData> From<D> for Tensor<D> {
    #[inline]
    fn from(data: D) -> Self {
        Self::new(data)
    }
}

impl<D: TensorData> AsRef<D> for Tensor<D> {
    #[inline]
    fn as_ref(&self) -> &D {
        &self.data
    }
}

impl<D: TensorData> AsMut<D> for Tensor<D> {
    #[inline]
    fn as_mut(&mut self) -> &mut D {
        &mut self.data
    }
}
