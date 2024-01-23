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
mod serde_wrapper;
mod shape;
mod shape_data;

pub use aliases::*;
pub use data::{tensor, TensorData};
pub use element::{Element, Float, MoreNumOps, Num};
pub use owned::Tensor;
pub use shape::{Len, Shape};
use std::mem;

impl<X: Num, const LEN: usize> vector<X, LEN>
where [(); LEN]: Len<LEN>
{
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

impl<X: Num, const LEN: usize> Vector<X, LEN>
where [(); LEN]: Len<LEN>
{
    /// Adds the [`Vector`]s `self` and `rhs`.
    pub fn add_vec(mut self, rhs: &Self) -> Self {
        self.iter_elem_mut().zip(rhs.iter_elem()).for_each(|(l, r)| *l += *r);
        self
    }

    /// Calculates `self * other^T`
    pub fn span_mat<const LEN2: usize>(&self, other: &vector<X, LEN2>) -> Matrix<X, LEN2, LEN>
    where
        [[(); LEN2]; LEN]: Len<{ LEN2 * LEN }>,
        [(); LEN2]: Len<LEN2>,
    {
        let mut mat = Matrix::zeros();
        for (row, &y) in self.iter_elem().enumerate() {
            for (col, &x) in other.iter_elem().enumerate() {
                mat[row][col].set(x * y)
            }
        }
        mat
    }
}

pub fn span_mat<X: Num, const LEN: usize, const LEN2: usize>(
    s: &vector<X, LEN>,
    other: &vector<X, LEN2>,
) -> matrix<X, LEN2, LEN>
where
    [(); LEN]: Len<LEN>,
    [(); LEN2]: Len<LEN2>,
{
    let mut mat = matrix::new([[X::ZERO; LEN2]; LEN]);
    for (row, &y) in s.iter_elem().enumerate() {
        for (col, &x) in other.iter_elem().enumerate() {
            mat[row][col].set(x * y);
        }
    }
    mat
}

impl<X: Num, const W: usize, const H: usize> matrix<X, W, H>
where [(); W]: Len<W>
{
    /// Multiplies the [`matrix`] `self` by the [`vector`] `vec` and returns a newly allocated
    /// [`Vector`] containing the result.
    pub fn mul_vec(&self, vec: &vector<X, W>) -> Vector<X, H> {
        let mut out = Vector::new([X::ZERO; H]);
        for (row, out) in self.iter_sub_tensors().zip(out.iter_sub_tensors_mut()) {
            out.set(row.dot_product(vec));
        }
        out
    }
}

impl<X: Num, const W: usize, const H: usize> Matrix<X, W, H> {
    /// Transposes the [`Matrix`].
    pub fn transpose<const LEN: usize>(self) -> Matrix<X, H, W>
    where [[(); H]; W]: Len<LEN> {
        let mut transposed = Matrix::<X, H, W>::new_uninit(); // bench vs zeros
        for y in 0..H {
            for x in 0..W {
                transposed[x][y].0.write(self[y][x].0);
            }
        }
        unsafe { mem::transmute(transposed) }
    }
}

impl<X: Num, const LEN: usize> vector<X, LEN> {
    /// Transmutes the [`vector`] as a [`matrix`] with height equal to `1`.
    pub fn as_row_mat(&self) -> &matrix<X, LEN, 1>
    where
        [(); LEN]: Len<LEN>,
        [[(); LEN]; 1]: Len<LEN>,
    {
        self.transmute_as()
    }

    /// Transmutes the [`vector`] as a [`matrix`] with width equal to `1`.
    pub fn as_col_mat(&self) -> &matrix<X, 1, LEN>
    where
        [(); LEN]: Len<LEN>,
        [[(); 1]; LEN]: Len<LEN>,
    {
        self.transmute_as()
    }
}

#[cfg(test)]
mod tests;
