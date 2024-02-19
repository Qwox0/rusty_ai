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

mod aliases;
mod data;
mod element;
mod maybe_uninit;
mod multidim_arr;
mod multidimensional;
mod owned;

pub use aliases::*;
pub use data::tensor;
pub use element::{Element, Float, MoreNumOps, Num};
pub use multidim_arr::{Len, MultidimArr};
pub use multidimensional::{Multidimensional, MultidimensionalOwned};
pub use owned::Tensor;
use std::mem;

/// trait alias for `MultidimArr<Element = ()>`.
pub trait Shape: MultidimArr<Element = ()> {}
impl<S: MultidimArr<Element = ()>> Shape for S {}

impl<X: Num, const LEN: usize> vector<X, LEN> {
    /// Calculates the dot product of the [`vector`]s `self` and `other`.
    /// <https://en.wikipedia.org/wiki/Dot_product>
    pub fn dot_product(&self, other: &Self) -> X {
        let mut res = X::ZERO;
        for (a, b) in self.iter_elem().zip(other.iter_elem()) {
            res += *a * *b;
        }
        res
    }
}

impl<X: Num, const LEN: usize> Vector<X, LEN> {
    /// Adds the [`Vector`]s `self` and `rhs`.
    pub fn add_vec(mut self, rhs: &Self) -> Self {
        self.iter_elem_mut().zip(rhs.iter_elem()).for_each(|(l, r)| *l += *r);
        self
    }

    /// Calculates `self * other^T`
    pub fn span_mat<const LEN2: usize>(&self, other: &vector<X, LEN2>) -> Matrix<X, LEN2, LEN> {
        let mut mat = Matrix::<X, LEN2, LEN>::new_uninit();
        for (row, &y) in self.iter_elem().enumerate() {
            for (col, &x) in other.iter_elem().enumerate() {
                mat[row][col].0.write(x * y);
            }
        }
        unsafe { mem::transmute(mat) }
    }
}

impl<X: Num, const W: usize, const H: usize> matrix<X, W, H> {
    /// Multiplies the [`matrix`] `self` by the [`vector`] `vec` and returns a newly allocated
    /// [`Vector`] containing the result.
    pub fn mul_vec(&self, vec: &vector<X, W>) -> Vector<X, H> {
        let mut out = Vector::<X, H>::new_uninit();
        for (row, out) in self.iter_sub_tensors().zip(out.iter_elem_mut()) {
            out.write(row.dot_product(vec));
        }
        unsafe { mem::transmute(out) }
    }
}

impl<X: Num, const W: usize, const H: usize> Matrix<X, W, H> {
    /// Transposes the [`Matrix`].
    pub fn transpose(self) -> Matrix<X, H, W> {
        let mut transposed = Matrix::<X, H, W>::new_uninit(); // bench vs zeros
        for y in 0..H {
            for x in 0..W {
                transposed[x][y].0.write(self[y][x].0);
            }
        }
        unsafe { mem::transmute(transposed) }
    }
}

impl<X: Num, const N: usize> Matrix<X, N, N> {
    /// Creates a new identity [`Matrix`] of dimension `N`.
    pub fn identity() -> Matrix<X, N, N> {
        let mut id = Matrix::<X, N, N>::zeros();
        for idx in 0..N {
            id[idx][idx].0 = X::ONE;
        }
        unsafe { mem::transmute(id) }
    }
}

impl<X: Num, const LEN: usize> vector<X, LEN> {
    /// Transmutes the [`vector`] as a [`matrix`] with height equal to `1`.
    pub fn as_row_mat(&self) -> &matrix<X, LEN, 1>
    where
        [(); LEN]: Shape + Len<LEN>,
        [[(); LEN]; 1]: Shape + Len<LEN>,
    {
        self.transmute_as()
    }

    /// Transmutes the [`vector`] as a [`matrix`] with width equal to `1`.
    pub fn as_col_mat(&self) -> &matrix<X, 1, LEN>
    where
        [(); LEN]: Shape + Len<LEN>,
        [[(); 1]; LEN]: Shape + Len<LEN>,
    {
        self.transmute_as()
    }
}

#[cfg(test)]
mod tests;
