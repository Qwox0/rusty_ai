#![feature(type_alias_impl_trait)]
#![feature(impl_trait_in_assoc_type)]
#![feature(int_roundings)]
#![feature(non_null_convenience)]
#![feature(ptr_sub_ptr)]
#![feature(exact_size_is_empty)]

use iter_rows::{IterRows, IterRowsMut};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Write},
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

pub mod iter_rows;

mod number;
pub use number::*;

mod ring;
pub use ring::*;

mod element;
pub use element::{Element, Float, Num};

mod util;
use util::*;

#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct Matrix<X> {
    width: usize,
    /// `height * width` is stored in `elements.len()`
    elements: Box<[X]>,
}

impl<X> Default for Matrix<X> {
    /// Creates a new 0x0 [`Matrix`].
    fn default() -> Self {
        Self { width: 0, elements: Box::new([]) }
    }
}

impl<X: Element> Matrix<X> {
    /// Creates a new [`Matrix`].
    ///
    /// # Safety
    ///
    /// `elements.len()` must be a multiple of `width`. Otherwise there might be problems with
    /// `Matrix::iter_rows` and `Matrix::iter_rows_mut`.
    #[inline]
    pub const unsafe fn new_unchecked(width: usize, elements: Box<[X]>) -> Self {
        Self { width, elements }
    }

    /// Creates a new [`Matrix`] if `elements.len()` is a multiple of `width`.
    pub fn new(width: usize, elements: Box<[X]>) -> Option<Self> {
        let len = elements.len();
        if len.checked_rem(width).unwrap_or(len) == 0 {
            Some(unsafe { Self::new_unchecked(width, elements) })
        } else {
            None
        }
    }

    /// Creates a [`Matrix`] from its dimensions and a value [`Iterator`].
    /// The [`Matrix`] is filled with values row by row.
    ///
    /// # Panics
    ///
    /// * Panics if the iterator is too small.
    /// * If `width * height` overflows.
    pub fn from_iter(width: usize, height: usize, iter: impl IntoIterator<Item = X>) -> Matrix<X> {
        let len = width * height;
        let elements: Box<[X]> = iter.into_iter().take(len).collect();
        assert_eq!(elements.len(), len);
        unsafe { Matrix::new_unchecked(width, elements) }
    }

    /// Creates a [`Matrix`] from its dimensions and a value [`Iterator`] and returns [`None`] if
    /// * the iterator is too small.
    /// * `width * height` overflows.
    ///
    /// The [`Matrix`] is filled with values row by row.
    ///
    /// ```rust
    /// # use matrix::Matrix;
    /// let m = Matrix::checked_from_iter(2, 2, 1..).expect("could create matrix");
    /// assert_eq!(m.get_row(0), Some([1, 2].as_slice()));
    /// assert_eq!(m.get_row(1), Some([3, 4].as_slice()));
    /// assert_eq!(m.get_row(2), None);
    /// ```
    pub fn checked_from_iter(
        width: usize,
        height: usize,
        iter: impl IntoIterator<Item = X>,
    ) -> Option<Matrix<X>> {
        let len = width.checked_mul(height)?;
        let elements: Box<[X]> = iter.into_iter().take(len).collect();
        if elements.len() != len {
            return None;
        }
        unsafe { Some(Matrix::new_unchecked(width, elements)) }
    }

    /// Creates a [`Matrix`] from an Iterator of rows.
    ///
    /// ```rust
    /// # use matrix::Matrix;
    /// Matrix::from_rows([[1, 0].as_slice(), [0, 1].as_slice()]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the rows don't have the same length:
    ///
    /// ```rust, should_panic
    /// # use matrix::Matrix;
    /// Matrix::from_rows([[1, 0].as_slice(), [0].as_slice()]); // -> Panics
    /// ```
    pub fn from_rows<'a>(elements: impl IntoIterator<Item = &'a [X]>) -> Matrix<X>
    where X: 'a + Clone {
        let mut elements_iter = elements.into_iter();
        let Some(first) = elements_iter.next() else {
            return Matrix::default();
        };
        let width = first.len();

        let mut elements = Vec::with_capacity(width * (1 + elements_iter.size_hint().0));
        elements.extend_from_slice(first);
        while let Some(row) = elements_iter.next() {
            if row.len() != width {
                panic!("row length must equal width (width: {width}, got: {})", row.len());
            }
            elements.extend_from_slice(row);
        }
        unsafe { Matrix::new_unchecked(width, elements.into_boxed_slice()) }
    }

    pub fn with_default(width: usize, height: usize, default: X) -> Matrix<X> {
        let elements = vec![default; width * height];
        unsafe { Matrix::new_unchecked(width, elements.into_boxed_slice()) }
    }
}

impl<X> Matrix<X> {
    /// Gets the width of the [`Matrix`].
    #[doc(alias = "get_input_count")]
    #[inline]
    pub const fn get_width(&self) -> usize {
        self.width
    }

    /// Gets the height of the [`Matrix`].
    ///
    /// # Panics
    ///
    /// Panics if `self.width == 0` and `self.elements.len() > 0`
    #[doc(alias = "get_neuron_count")]
    #[inline]
    pub const fn get_height(&self) -> usize {
        match self.elements.len().checked_div(self.width) {
            Some(height) => height,
            None if self.elements.len() == 0 => 0,
            None => panic!("`self.width` should be greater than `0`"),
        }
    }

    /// Gets the elements of the [`Matrix`] as a slice.
    pub const fn get_elements(&self) -> &[X] {
        &self.elements
    }

    /// Gets the elements of the [`Matrix`] as a mutable slice.
    pub fn get_elements_mut(&mut self) -> &mut [X] {
        &mut self.elements
    }

    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&[X]> {
        self.elements.get(y * self.width..(y + 1) * self.width)
    }

    #[inline]
    pub fn get_row_mut(&mut self, y: usize) -> Option<&mut [X]> {
        self.elements.get_mut(y * self.width..(y + 1) * self.width)
    }

    #[inline]
    pub fn get(&self, y: usize, x: usize) -> Option<&X> {
        self.elements.get(y * self.width + x)
    }

    #[inline]
    pub fn get_mut(&mut self, y: usize, x: usize) -> Option<&mut X> {
        self.elements.get_mut(y * self.width + x)
    }

    #[inline]
    pub const fn iter_rows(&self) -> IterRows<'_, X> {
        // SAFETY: `self.elements.len()` is a multiple of `self.width`. See [`IterRows::new`]
        unsafe { IterRows::new(&self.elements, self.width) }
    }

    #[inline]
    pub fn iter_rows_mut(&mut self) -> IterRowsMut<'_, X> {
        unsafe { IterRowsMut::new(&mut self.elements, self.width) }
    }

    pub fn iter<'a>(&'a self) -> Iter<'_, X> {
        self.elements.iter()
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'_, X> {
        self.elements.iter_mut()
    }

    /// (width, height)
    pub const fn get_dimensions(&self) -> (usize, usize) {
        (self.get_width(), self.get_height())
    }
}

impl<X: Element, const W: usize, const H: usize> From<[[X; W]; H]> for Matrix<X> {
    fn from(elements: [[X; W]; H]) -> Self {
        let elements = elements.into_iter().flatten().collect();
        unsafe { Matrix::new_unchecked(W, elements) }
    }
}

impl<X> IntoIterator for Matrix<X> {
    type IntoIter = std::vec::IntoIter<X>;
    type Item = X;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_vec().into_iter()
    }
}

impl<'a, X> IntoIterator for &'a Matrix<X> {
    type IntoIter = Iter<'a, X>;
    type Item = &'a X;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<'a, X> IntoIterator for &'a mut Matrix<X> {
    type IntoIter = IterMut<'a, X>;
    type Item = &'a mut X;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter_mut()
    }
}

impl<X: Num> Matrix<X> {
    pub fn with_zeros(width: usize, height: usize) -> Matrix<X> {
        Matrix::with_default(width, height, X::zero())
    }

    /// Creates the `n` by `n` identity Matrix.
    pub fn identity(n: usize) -> Matrix<X> {
        (0..n).into_iter().fold(Matrix::with_zeros(n, n), |mut acc, i| {
            *acc.get_mut(i, i).unwrap() = X::one();
            acc
        })
    }
}

impl<X: Num> Matrix<X> {
    pub fn mul_vec(&self, vec: &[X]) -> Vec<X> {
        assert_eq!(
            self.width,
            vec.len(),
            "Vector has incompatible dimensions (expected: {}, got: {})",
            self.width,
            vec.len()
        );
        self.iter_rows().map(|row| dot_product_unchecked(row, vec)).collect()
    }
}

macro_rules! impl_mul {
    ( $( $matrix: ty )*) => { $(
        impl<X: Num, V> std::ops::Mul<V> for $matrix
        where
            V: AsRef<[X]>,
        {
            type Output = Vec<X>;

            fn mul(self, rhs: V) -> Self::Output {
                self.mul_vec(rhs.as_ref())
            }
        }
    )* };
}

impl_mul! { Matrix<X> &Matrix<X> }

impl<X> Index<(usize, usize)> for Matrix<X> {
    type Output = X;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).unwrap()
    }
}

impl<X> IndexMut<(usize, usize)> for Matrix<X> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1).unwrap()
    }
}

impl<X: Display> Matrix<X> {
    pub fn to_string_with_title(&self, title: impl ToString) -> Result<String, std::fmt::Error> {
        let title = title.to_string();
        let mut title = title.trim();

        let column_widths = self.iter_rows().fold(vec![0; self.width], |acc, row| {
            row.iter().zip(acc).map(|(e, max)| e.to_string().len().max(max)).collect()
        });

        let content_width = column_widths.len() + column_widths.iter().sum::<usize>() - 1;
        let content = self
            .iter_rows()
            .map(|row| {
                row.iter()
                    .zip(column_widths.iter())
                    .map(|(t, width)| format!("{t:^width$}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .map(|row_str| format!("│ {row_str} │"))
            .collect::<Vec<_>>()
            .join("\n");

        let title_len = title.chars().count();
        if title_len > content_width {
            title = title.split_at(content_width).0
        }

        let pad_width = content_width.checked_sub(title_len).unwrap_or(0);
        let pre_title_pad = " ".repeat(pad_width.div_floor(2));
        let post_title_pad = " ".repeat(pad_width.div_ceil(2));

        let mut buf = String::with_capacity((content_width + 4) * (self.get_height() + 2));

        writeln!(&mut buf, "┌ {}{}{} ┐", pre_title_pad, title, post_title_pad)?;
        writeln!(&mut buf, "{}", content)?;
        write!(&mut buf, "└ {} ┘", " ".repeat(content_width))?;
        Ok(buf)
    }
}

impl<X: Display> Display for Matrix<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = self.to_string_with_title("")?;
        write!(f, "{}", text)
    }
}

/*
#[cfg(test)]
mod tests {
    use crate::matrix::SimdMatrix;

    use super::Matrix;

    #[test]
    fn identity() {
        let res: Matrix<f64> = Matrix::identity(3);
        assert_eq!(
            res.elements,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0]
            ]
        );
    }

    /*
    const M: usize = 11;
    const N: usize = M - 1;
    const K: usize = M - 10;
    */

    const M: usize = 512;
    const N: usize = M;
    const K: usize = M;

    #[test]
    fn bench() {
        let a = Matrix::new_random(K, M);
        let b = Matrix::new_random(N, K);
        let mut c = Matrix::with_zeros(N, M);

        let start = std::time::Instant::now();
        c.matrix_mul(&a, &b);
        c.matrix_mul(&a, &b);
        let secs = start.elapsed().as_secs_f64() / 2.0;
        println!(
            "1: {} GFLOPS/s",
            ((2 * M * N * K) as f64 / secs as f64) / 1e9
        );
        //println!("{:.2?}", c);

        let c1 = c.clone().elements.concat();

        let a = SimdMatrix::new(a);
        let b = SimdMatrix::new(b);
        let mut c = SimdMatrix::new(Matrix::with_zeros(N, M));

        let start = std::time::Instant::now();
        c.matrix_mul(&a, &b);
        c.matrix_mul(&a, &b);
        let secs = start.elapsed().as_secs_f64() / 2.0;
        println!(
            "2: {} GFLOPS/s",
            ((2 * M * N * K) as f64 / secs as f64) / 1e9
        );

        assert!(
            c1.iter()
                .zip(c.vec)
                .map(|(a, b)| a - b)
                .map(|x| x * x)
                .sum::<f64>()
                < 1e-7
        );
        let mut c = SimdMatrix::new(Matrix::with_zeros(N, M));

        let start = std::time::Instant::now();
        c.matrix_mul_simd(&a, &b);
        c.matrix_mul_simd(&a, &b);
        let secs = start.elapsed().as_secs_f64() / 2.0;
        println!(
            "SIMD: {} GFLOPS/s",
            ((2 * M * N * K) as f64 / secs as f64) / 1e9
        );
        //println!("{:.2?}", c);

        assert!(
            c1.iter()
                .zip(c.vec)
                .map(|(a, b)| a - b)
                .map(|x| x * x)
                .sum::<f64>()
                < 1e-7
        );

        assert!(false)
    }
}

const NELTS: usize = 8;

impl Matrix<f64> {
    fn get_mut(&mut self, y: usize, x: usize) -> &mut f64 {
        &mut self.elements[y][x]
    }

    fn matrix_mul(&mut self, a: &Matrix<f64>, b: &Matrix<f64>) {
        for m in 0..self.get_height() {
            for n in 0..self.get_width() {
                for k in 0..a.get_width() {
                    *self.get_mut(m, n) += a.get(m, k).expect("1") * b.get(k, n).expect("2")
                }
            }
        }
    }
}

#[derive(Debug)]
struct SimdMatrix {
    pub vec: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl SimdMatrix {
    fn new(matrix: Matrix<f64>) -> SimdMatrix {
        SimdMatrix {
            vec: matrix.elements.into_iter().concat(),
            width: matrix.width,
            height: matrix.height,
        }
    }

    fn get(&self, y: usize, x: usize) -> &f64 {
        &self.vec[y * self.width + x]
    }

    fn get_mut(&mut self, y: usize, x: usize) -> &mut f64 {
        &mut self.vec[y * self.width + x]
    }

    fn load_simd(&self, y: usize, x: usize) -> std::simd::Simd<f64, NELTS> {
        let a = y * self.width + x;
        std::simd::Simd::from_slice(&self.vec[a..a + NELTS])
    }

    fn load_simd_tr(&self, y: usize, x: usize) -> std::simd::Simd<f64, NELTS> {
        let start = y * self.width + x;
        let mut column: std::simd::f64x8 = std::simd::Simd::splat(0.0);
        for (idx, val) in self
            .vec
            .iter()
            .skip(start)
            .step_by(self.width)
            .take(NELTS)
            .enumerate()
        {
            column[idx] = *val;
        }
        column
    }

    fn load_simd_tr2(&self, y: usize, x: usize) -> std::simd::Simd<f64, NELTS> {
        let w = self.width;
        let start = y * self.width + x;
        let slice = &self.vec[start..];
        let idxs = std::simd::Simd::from_array([0, w, 2 * w, 3 * w, 4 * w, 5 * w, 6 * w, 7 * w]);
        std::simd::Simd::gather_or_default(slice, idxs)
    }

    fn matrix_mul(&mut self, a: &SimdMatrix, b: &SimdMatrix) {
        for m in 0..self.height {
            for n in 0..self.width {
                for k in 0..a.width {
                    *self.get_mut(m, n) += a.get(m, k) * b.get(k, n);
                }
            }
        }
    }

    fn matrix_mul_simd(&mut self, a: &SimdMatrix, b: &SimdMatrix) {
        let edge = NELTS * (a.width / NELTS);
        for m in 0..self.height {
            for n in 0..self.width {
                let mut tmp = std::simd::Simd::from_array([0.0; NELTS]);
                for kv in (0..edge).step_by(NELTS) {
                    tmp += a.load_simd(m, kv) * b.load_simd_tr2(kv, n);
                }
                *self.get_mut(m, n) += tmp.reduce_sum();
                for k in edge..a.width {
                    *self.get_mut(m, n) += a.get(m, k) * b.get(k, n);
                }
            }
        }
    }

}
*/
