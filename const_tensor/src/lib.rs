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

use std::{
    borrow::{Borrow, BorrowMut},
    ops::{Add, Deref, DerefMut, IndexMut},
};

mod data;
mod element;
mod macros;
mod tensor;

use data::AsArr;
pub use data::{Len, TensorData};
pub use element::{Element, Float, MoreNumOps, Num};
use macros::{count, make_tensor, ArrDefault};
pub use tensor::Tensor;

make_tensor! { Scalar scalar : => X, Sub: Self }
make_tensor! { Vector vector : LEN => [X; LEN], Sub: scalar<X> }
make_tensor! { Matrix matrix : W H => [[X; W]; H], Sub: vector<X, W> }
make_tensor! { Tensor3 tensor3: A B C => [[[X; A]; B]; C], Sub: matrix<X, A, B> }
make_tensor! { Tensor4 tensor4: A B C D => [[[[X; A]; B]; C]; D], Sub: tensor3<X, A, B, C> }
make_tensor! {
    Tensor5 tensor5: A B C D E => [[[[[X; A]; B]; C]; D]; E],
    Sub: tensor4<X, A, B, C, D>
}
make_tensor! {
    Tensor6 tensor6: A B C D E F => [[[[[[X; A]; B]; C]; D]; E]; F],
    Sub: tensor5<X, A, B, C, D, E>
}
make_tensor! {
    Tensor7 tensor7: A B C D E F G => [[[[[[[X; A]; B]; C]; D]; E]; F]; G],
    Sub: tensor6<X, A, B, C, D, E, F>
}

impl<X: Num, const LEN: usize> vector<X, LEN>
where Self: Len<LEN>
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
where vector<X, LEN>: Len<LEN>
{
    /// Adds the [`Vector`]s `self` and `rhs`.
    pub fn add_vec(mut self, rhs: &Self) -> Self {
        self.iter_elem_mut().zip(rhs.iter_elem()).for_each(|(l, r)| *l += *r);
        self
    }

    /// Calculates `self * other^T`
    pub fn span_mat<const LEN2: usize>(&self, other: &vector<X, LEN2>) -> Matrix<X, LEN2, LEN>
    where
        matrix<X, LEN2, LEN>: Len<{ LEN2 * LEN }>,
        vector<X, LEN2>: Len<LEN2>,
    {
        let mut mat = Matrix::zeros();
        for (row, &y) in self.iter_elem().enumerate() {
            for (col, &x) in other.iter_elem().enumerate() {
                mat._as_inner_mut()[row][col] = x * y
            }
        }
        mat
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
    where matrix<X, H, W>: Len<LEN> {
        let inner = self._as_inner().as_arr();
        let mut transposed = Matrix::zeros();
        let transposed_inner = transposed._as_inner_mut();
        for (y, row) in inner.iter().enumerate() {
            for (x, el) in row.iter().enumerate() {
                transposed_inner[x][y] = *el
            }
        }
        transposed
    }
}

impl<X: Num, const LEN: usize> vector<X, LEN> {
    /// Transmutes the [`vector`] as a [`matrix`] with height equal to `1`.
    pub fn as_row_mat(&self) -> &matrix<X, LEN, 1>
    where
        Self: Len<LEN>,
        matrix<X, LEN, 1>: Len<LEN>,
    {
        self.transmute_as()
    }

    /// Transmutes the [`vector`] as a [`matrix`] with width equal to `1`.
    pub fn as_col_mat(&self) -> &matrix<X, 1, LEN>
    where
        Self: Len<LEN>,
        matrix<X, 1, LEN>: Len<LEN>,
    {
        self.transmute_as()
    }
}

#[cfg(test)]
mod tests;
