//! # Const tensor crate
//!
//! Implements [`Tensor`] trait for `Box<[[[[X; A]; B]; ...]; Z]>` of any dimensions.
//!
//! # [`private::Sealed`] Note
//!
//! This crate contains unsafe Code. To ensure Safety only the element traits can be implemented
//! for custom types

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![warn(missing_docs)]

use core::slice;
use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
    iter::Map,
    mem,
    ops::{Add, Deref, DerefMut},
    usize,
};

mod element;
pub use element::*;
use inline_closure::inline_closure;

/*
   mod interface;
//pub use interface::TensorI;

mod container;
pub use container::*;

mod data;
pub use data::*;

mod shape;
pub use shape::*;
*/

pub unsafe trait Tensor<X: Element>:
    Sized
    + Debug
    + Deref<Target = Self::Data>
    + DerefMut
    + AsRef<Self::Data>
    + AsMut<Self::Data>
    + Borrow<Self::Data>
    + BorrowMut<Self::Data>
{
    type Data: TensorData<X, Owned = Self>;

    /// Creates a new Tensor.
    fn from_box(data: Box<Self::Data>) -> Self;

    /// Creates a new Tensor.
    #[inline]
    fn new(data: <Self::Data as TensorData<X>>::Shape) -> Self {
        Self::from_box(Self::Data::new_boxed(data))
    }

    /// Creates the Tensor from a 1D representation of its elements.
    #[inline]
    fn from_1d(vec: Vector<X, { Self::Data::LEN }>) -> Self {
        let vec = mem::ManuallyDrop::new(vec);
        unsafe { mem::transmute_copy(&vec) }
    }

    /// Converts the Tensor into the 1D representation of its elements.
    #[inline]
    fn into_1d<const LEN: usize>(self) -> Vector<X, LEN>
    where Self::Data: Len<LEN> {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }

    /// Changes the Shape of the Tensor.
    ///
    /// The generic constants ensure
    #[inline]
    fn transmute_into<T2: Tensor<X>, const LEN: usize>(self) -> T2
    where
        Self::Data: Len<LEN>,
        T2::Data: Len<LEN>,
    {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }
    /*
    fn transmute_into<T2: Tensor<X>>(self) -> T2
    where T2::Data: Len<{ Self::Data::LEN }> {
        let tensor = mem::ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&tensor) }
    }
    */

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem_mut<const LEN: usize>(mut self, f: impl FnMut(&mut X)) -> Self
    where Self::Data: Len<LEN> {
        self.iter_elem_mut().for_each(f);
        self
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_elem<const LEN: usize>(self, mut f: impl FnMut(X) -> X) -> Self
    where Self::Data: Len<LEN> {
        self.map_elem_mut(|x| *x = f(*x))
    }
}

pub unsafe trait Len<const LEN: usize> {}

unsafe impl<X: Element> Len<1> for X {}

pub unsafe trait TensorData<X: Element>: Sized {
    type Owned: Tensor<X, Data = Self>;
    type Shape: Copy;
    type SubData: TensorData<X>;

    const LEN: usize;
    const SUB_COUNT: usize;

    fn new_boxed(data: Self::Shape) -> Box<Self>;

    fn into_inner(self) -> Self::Shape;
    fn as_inner(&self) -> &Self::Shape;
    fn as_inner_mut(&mut self) -> &mut Self::Shape;

    fn iter_inner(&self) -> slice::Iter<'_, <Self::SubData as TensorData<X>>::Shape>;
    fn iter_inner_mut(&mut self) -> slice::IterMut<'_, <Self::SubData as TensorData<X>>::Shape>;

    fn literal<'a>(data: Self::Shape) -> &'a Self {
        Box::leak(Self::new_boxed(data))
    }

    fn wrap_ref(data: &Self::Shape) -> &Self {
        unsafe { mem::transmute(data) }
    }

    fn wrap_ref_mut(data: &mut Self::Shape) -> &mut Self {
        unsafe { mem::transmute(data) }
    }

    /// Clones `self` into a new [`Box`].
    fn to_box(&self) -> Box<Self> {
        //Box::new(tensor { data: self.data.clone() });
        Self::new_boxed(self.as_inner().clone())
    }

    fn set(&mut self, val: Self::Shape) {
        *self.as_inner_mut() = val;
    }

    /// Changes the Shape of the Tensor.
    #[inline]
    fn transmute_as<T2: TensorData<X> + Len<{ Self::LEN }>>(&self) -> &T2 {
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over references to the sub tensors of the tensor.
    fn iter_sub_tensors<'a>(
        &'a self,
    ) -> Map<
        slice::Iter<'a, <Self::SubData as TensorData<X>>::Shape>,
        impl Fn(&'a <Self::SubData as TensorData<X>>::Shape) -> &'a Self::SubData,
    > {
        self.iter_inner().map(Self::SubData::wrap_ref)
    }

    /// Creates an [`Iterator`] over mutable references to the sub tensors of the tensor.
    fn iter_sub_tensors_mut<'a>(
        &'a mut self,
    ) -> Map<
        slice::IterMut<'_, <Self::SubData as TensorData<X>>::Shape>,
        impl Fn(&'a mut <Self::SubData as TensorData<X>>::Shape) -> &'a mut Self::SubData,
    > {
        self.iter_inner_mut().map(Self::SubData::wrap_ref_mut)
    }
}

pub trait TensorDataWithLen<X: Element, const LEN: usize>: TensorData<X> + Len<LEN> {
    /// Creates a reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d(&self) -> &vector<X, LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates a mutable reference to the elements of the tensor in its 1D representation.
    #[inline]
    fn as_1d_mut(&mut self) -> &mut vector<X, LEN> {
        // TODO: test
        unsafe { mem::transmute(self) }
    }

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    #[inline]
    fn iter_elem(&self) -> slice::Iter<'_, X> {
        self.as_1d().as_inner().iter()
    }

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    #[inline]
    fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X> {
        self.as_1d_mut().as_inner_mut().iter_mut()
    }
}

impl<X: Element, T: TensorData<X> + Len<LEN>, const LEN: usize> TensorDataWithLen<X, LEN> for T {}

macro_rules! rep_default {
    (; $default:expr) => {
        $default
    };
    ($val:expr; $default:expr) => {
        $val
    };
}

macro_rules! make_tensor {
    (
        $name:ident $data_name:ident : $($dim_name:ident)* => $shape:ty,
        Sub: $sub_data:ty : $sub_count:expr
        $(,iter_inner: $iter_inner:expr)?
        $(,iter_inner_mut: $iter_inner_mut:expr)?
        $(,)?
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        pub struct $data_name<X: Element, $( const $dim_name: usize ),*> {
            data: $shape
        }

        unsafe impl<X: Element, $( const $dim_name: usize ),*> Len<{ $($dim_name * )* 1 }> for $data_name<X, $( $dim_name ),*> {}

        unsafe impl<X: Element, $( const $dim_name: usize ),*> TensorData<X> for $data_name<X, $( $dim_name ),*> {
            type Owned = $name<X, $( $dim_name ),*>;
            type Shape = $shape;
            type SubData = $sub_data;

            const LEN: usize = $($dim_name * )* 1;
            const SUB_COUNT: usize = $sub_count;

            #[inline]
            fn new_boxed(data: Self::Shape) -> Box<Self> { Box::new(Self { data }) }

            #[inline]
            fn into_inner(self) -> Self::Shape { self.data }
            #[inline]
            fn as_inner(&self) -> &Self::Shape { &self.data }
            #[inline]
            fn as_inner_mut(&mut self) -> &mut Self::Shape { &mut self.data }

            #[inline]
            fn iter_inner(&self) -> slice::Iter<'_, <Self::SubData as TensorData<X>>::Shape> {
                rep_default!(
                    $(inline_closure!($iter_inner))? ;
                    { self.data.iter() }
                )
            }
            #[inline]
            fn iter_inner_mut(&mut self) -> slice::IterMut<'_, <Self::SubData as TensorData<X>>::Shape> {
                rep_default!(
                    $(inline_closure!($iter_inner_mut))? ;
                    { self.data.iter_mut() }
                )
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name<X: Element, $( const $dim_name: usize ),*> {
            data: Box<$data_name<X, $( $dim_name ),*>>,
        }

        unsafe impl<X: Element, $( const $dim_name: usize ),*> Tensor<X> for $name<X, $( $dim_name ),*> {
            type Data = $data_name<X, $( $dim_name ),*>;

            #[inline]
            fn from_box(data: Box<Self::Data>) -> Self {
                Self { data }
            }
        }

        impl<X: Element, $( const $dim_name: usize ),*> Deref for $name<X, $( $dim_name ),*> {
            type Target = $data_name<X, $( $dim_name ),*>;

            #[inline]
            fn deref(&self) -> &Self::Target { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> DerefMut for $name<X, $( $dim_name ),*> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> AsRef<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn as_ref(&self) -> &$data_name<X, $( $dim_name ),*> { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> AsMut<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn as_mut(&mut self) -> &mut $data_name<X, $( $dim_name ),*> { &mut self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> Borrow<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn borrow(&self) -> &$data_name<X, $( $dim_name ),*> { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> BorrowMut<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn borrow_mut(&mut self) -> &mut $data_name<X, $( $dim_name ),*> { &mut self.data }
        }
    };
}

make_tensor! { Scalar scalar : => X, Sub: Self : 1,
iter_inner: |self| unsafe { mem::transmute::<&Self, &[X; 1]>(self) }.iter(),
iter_inner_mut: |self| unsafe { mem::transmute::<&mut Self, &mut [X; 1]>(self) }.iter_mut()
}

make_tensor! { Vector vector : LEN => [X; LEN], Sub: scalar<X> : LEN }
make_tensor! { Matrix matrix : W H => [[X; W]; H], Sub: vector<X, W> : H }
make_tensor! { Tensor3 tensor3: A B C => [[[X; A]; B]; C], Sub: matrix<X, A, B> : B }
make_tensor! { Tensor4 tensor4: A B C D => [[[[X; A]; B]; C]; D], Sub: tensor3<X, A, B, C> : C }

// =====

impl<X: Num, const LEN: usize> vector<X, LEN>
where Self: Len<LEN>
{
    pub fn dot_product(&self, other: &Self) -> X {
        let mut res = X::ZERO;
        for (a, b) in self.iter_elem().zip(other.iter_elem()) {
            res += *a * *b;
        }
        res
    }
}

impl<X: Num, const LEN: usize> Vector<X, LEN>
where vector<X, LEN>: Len<LEN>
{
    pub fn add_vec(mut self, rhs: &Self) -> Self {
        self.iter_elem_mut().zip(rhs.iter_elem()).for_each(|(l, r)| *l += *r);
        self
    }
}

impl<X: Num, const LEN: usize> Add<&Self> for Vector<X, LEN>
where vector<X, LEN>: Len<LEN>
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self.add_vec(rhs)
    }
}

impl<X: Num, const W: usize, const H: usize> matrix<X, W, H>
where vector<X, W>: Len<W>
{
    pub fn mul_vec(&self, vec: impl Borrow<vector<X, W>>) -> Vector<X, H> {
        let mut out = Vector::new([X::ZERO; H]);
        for (row, out) in self.iter_sub_tensors().zip(out.iter_sub_tensors_mut()) {
            out.set(row.dot_product(vec.borrow()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        println!("\n# transmute_into:");
        let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat: Tensor3<i32, 3, 2, 2> = vec.transmute_into();
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# transmute_as:");
        let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let vec = vec.as_ref(); // Optional
        let mat = vec.transmute_as::<tensor3<i32, 3, 2, 2>>();
        println!("{:#?}", mat);
        assert_eq!(mat, tensor3::literal([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# from_1d:");
        let vec = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat = Tensor3::from_1d(vec);
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# iter components:");
        let mut iter = mat.iter_sub_tensors();
        assert_eq!(iter.next(), Some(matrix::literal([[1, 2, 3], [4, 5, 6]])));
        assert_eq!(iter.next(), Some(matrix::literal([[7, 8, 9], [10, 11, 12]])));
        assert_eq!(iter.next(), None);
        println!("works");

        println!("\n# iter components:");
        assert!(mat.iter_elem().enumerate().all(|(idx, elem)| *elem == 1 + idx as i32));
        println!("works");

        println!("\n# add one:");
        let mat = Matrix::new([[1i32, 2], [3, 4], [5, 6]]);
        let mat = mat.map_elem(|x| x + 1);
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[2, 3], [4, 5], [6, 7]]));

        println!("\n# dot_product:");
        let vec1 = Vector::new([1, 9, 2, 2]);
        let vec2 = Vector::new([1, 0, 5, 1]);
        let res = vec1.dot_product(vec2.as_ref());
        println!("{:#?}", res);
        assert_eq!(res, 13);

        println!("\n# mat_mul_vec:");
        let vec = Vector::new([2, 1]);
        let res = mat.mul_vec(vec);
        println!("{:#?}", res);
        assert_eq!(res, Tensor::new([7, 13, 19]));

        panic!("It works!");
    }
}
