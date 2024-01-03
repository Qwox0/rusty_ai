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
use std::{borrow::Borrow, iter::Map, mem, ops::Deref};

mod element;
pub use element::*;

mod interface;
//pub use interface::TensorI;

mod container;
pub use container::*;

mod data;
pub use data::*;

mod shape;
pub use shape::*;

impl<X: Num, const LEN: usize> tensor<[X; LEN], LEN> {
    pub fn dot_product(&self, other: &Self) -> X {
        let mut res = X::ZERO;
        for (a, b) in self.iter_elem().zip(other.iter_elem()) {
            res += *a * *b;
        }
        res
    }
}

impl<X: Num, const W: usize, const H: usize> tensor<[[X; W]; H], { W * H }>
where
    [X; W]: Shape + Len<W>,
    [X; H]: Shape + Len<H>,
{
    pub fn mul_vec(&self, vec: impl Borrow<VectorData<X, W>>) -> Vector<X, H> {
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
        let vec: Vector<i32, 12> = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat: Tensor3<i32, 3, 2, 2> = vec.transmute_into();
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# transmute_as:");
        let vec = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let vec = vec.as_ref(); // Optional
        let mat = vec.transmute_as::<[[[i32; 3]; 2]; 2]>();
        println!("{:#?}", mat);
        assert_eq!(mat, tensor::literal([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# from_1d:");
        let vec = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mat = Tensor3::from_1d(vec);
        println!("{:#?}", mat);
        assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

        println!("\n# iter components:");
        let mut iter = mat.iter_sub_tensors();
        assert_eq!(iter.next(), Some(tensor::literal([[1, 2, 3], [4, 5, 6]])));
        assert_eq!(iter.next(), Some(tensor::literal([[7, 8, 9], [10, 11, 12]])));
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
        let vec1 = Tensor::new([1, 9, 2, 2]);
        let vec2 = Tensor::new([1, 0, 5, 1]);
        let res = vec1.dot_product(vec2.as_ref());
        println!("{:#?}", res);
        assert_eq!(res, 13);

        println!("\n# mat_mul_vec:");
        let vec = Tensor::new([2, 1]);
        let res = mat.mul_vec(vec);
        println!("{:#?}", res);
        assert_eq!(res, Tensor::new([7, 13, 19]));

        panic!("It works!");
    }
}
