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

use std::{borrow::Borrow, ops::Deref};

mod element;
pub use element::*;

mod interface;
pub use interface::TensorI;

mod container;
pub use container::*;

mod data;
pub use data::*;

mod private {
    pub trait SealedData {}
    pub trait Sealed<X: crate::element::Element> {}
}

/// Type alias for a 0D tensor.
pub type Scalar<X> = Tensor<X>;
/// Type alias for a 1D tensor.
pub type Vector<X, const LEN: usize> = Tensor<[X; LEN]>;
/// Type alias for a 2D tensor.
pub type Matrix<X, const W: usize, const H: usize> = Tensor<[[X; W]; H]>;
/// Type alias for a 3D tensor.
pub type Tensor3<X, const A: usize, const B: usize, const C: usize> = Tensor<[[[X; A]; B]; C]>;
/// Type alias for a 4D tensor.
pub type Tensor4<X, const A: usize, const B: usize, const C: usize, const D: usize> =
    Tensor<[[[[X; A]; B]; C]; D]>;

trait VecOps {
    type Scalar;
    fn dot_product(&self, other: impl Deref<Target = Self>) -> Self::Scalar;
}

impl<X: Num, const LEN: usize> VecOps for VectorData<X, LEN> {
    type Scalar = X;

    fn dot_product(&self, other: impl Deref<Target = Self>) -> X {
        let mut res = X::ZERO;
        for (&a, &b) in self.iter_sub_tensors().zip(other.borrow().iter_sub_tensors()) {
            res += a * b;
        }
        res
    }
}

impl<X: Num, const W: usize, const H: usize> Matrix<X, W, H> {
    fn mul_vec(&self, vec: Vector<X, W>) -> Vector<X, H> {
        let mut out = Vector::new([X::ZERO; H]);
        for (row, out) in self.iter_sub_tensors().zip(out.iter_sub_tensors_mut()) {
            *out = vec_dot(row, vec.as_ref());
        }
        out
    }
}

/*
fn mat_mul_vec<X: Num, const W: usize, const H: usize>(
    mat: Matrix<X, W, H>,
    vec: Vector<X, W>,
) -> Vector<X, H> {
    let mut out = Vector::new([X::ZERO; H]);
    for (row, out) in mat.iter_sub_tensors().zip(out.iter_sub_tensors_mut()) {
        *out = vec_dot(row, vec.as_ref());
    }
    out
}
*/

fn vec_dot<X: Num, const N: usize>(vec1: &[X; N], vec2: &[X; N]) -> X {
    let mut res = X::ZERO;
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        res += a * b;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        println!("\n# transmute_into:");
        let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat = vec.transmute_into::<Tensor3<i32, 3, 2, 2>>();
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# transmute_as:");
        let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let vec = vec.as_ref(); // Optional
        let mat = vec.transmute_as::<Tensor3<i32, 3, 2, 2>>();
        println!("{:#?}", mat);
        assert_eq!(mat, &[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);

        println!("\n# from_1d:");
        let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat = Tensor3::from_1d(vec);
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# iter components:");
        let mut iter = mat.iter_sub_tensors();
        assert_eq!(iter.next(), Some(&[[1, 2, 3], [4, 5, 6]]));
        assert_eq!(iter.next(), Some(&[[7, 8, 9], [10, 11, 12]]));
        assert_eq!(iter.next(), None);
        println!("works");

        println!("\n# iter components:");
        assert!(mat.iter_elem().enumerate().all(|(idx, elem)| *elem == 1 + idx as i32));
        println!("works");

        println!("\n# add one:");
        let mat = Tensor::new([[1i32, 2], [3, 4], [5, 6]]);
        let mat = mat.map_elem(|x| x + 1);
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[2, 3], [4, 5], [6, 7]]));

        println!("\n# dot_product:");
        let vec1 = Vector::new([1, 9, 2, 2]);
        let vec2 = Vector::new([1, 0, 5, 1]);
        let res = vec1.dot_product(vec2);
        println!("{:#?}", res);
        assert_eq!(res, 13);

        println!("\n# mat_mul_vec:");
        let vec = Vector::new([2, 1]);
        let res = mat.mul_vec(vec);
        println!("{:#?}", res);
        assert_eq!(res, Tensor::new([7, 13, 19]));
    }
}
