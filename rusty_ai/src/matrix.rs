use crate::prelude::*;
use itertools::Itertools;
use rand::{distributions::DistIter, prelude::Distribution};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Write},
    ops::{Add, Index, IndexMut, Mul},
};

pub trait Ring: Sized + Add<Self, Output = Self> + Mul<Self, Output = Self> {
    const ZERO: Self;
    const ONE: Self;
}

macro_rules! impl_ring {
    ( $( $type:ty )+ : $zero:literal $one:literal ) => { $(
        impl Ring for $type {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
        }
    )+ };
}
impl_ring! { i8 i16 i32 i64 i128: 0 1 }
impl_ring! { u8 u16 u32 u64 u128: 0 1 }
impl_ring! { f32 f64: 0.0 1.0 }

#[derive(Debug, Clone, Default, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct Matrix<T: Sized> {
    width: usize,
    height: usize,
    elements: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    constructor! { new -> width: usize, height: usize, elements: Vec<Vec<T>> }

    impl_getter! { pub get_elements -> elements: &Vec<Vec<T>> }

    impl_getter! { pub get_elements_mut -> elements: &mut Vec<Vec<T>> }

    #[doc(alias = "get_input_count")]
    #[inline]
    pub fn get_width(&self) -> usize {
        self.width
    }

    #[doc(alias = "get_neuron_count")]
    #[inline]
    pub fn get_height(&self) -> usize {
        self.height
    }

    /// # Panics
    /// Panics if the iterator is too small.
    pub fn from_iter(width: usize, height: usize, iter: impl Iterator<Item = T>) -> Matrix<T> {
        let elements: Vec<_> =
            iter.chunks(width).into_iter().take(height).map(Iterator::collect).collect();
        assert_eq!(elements.len(), height);
        assert_eq!(elements.last().map(Vec::len), Some(width));
        Matrix::new(width, height, elements)
    }

    /// Create a [`Matrix`] from a [`Vec`] of Rows.
    /// ```rust
    /// # use rusty_ai::prelude::Matrix;
    /// Matrix::from_rows(vec![vec![1, 0], vec![0, 1]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the rows don't have the same length:
    /// ```rust, should_panic
    /// # use rusty_ai::prelude::Matrix;
    /// Matrix::from_rows(vec![vec![1, 0], vec![0]]); // -> Panics
    /// ```
    pub fn from_rows(elements: Vec<Vec<T>>) -> Matrix<T> {
        let height = elements.len();
        let width = elements.first().map(Vec::len).unwrap_or(0);
        assert!(elements.iter().map(Vec::len).all(|len| len == width));
        Matrix::new(width, height, elements)
    }

    /*
    /// Creates a [`Matrix`] containing an empty elements [`Vec`] with a
    /// capacity of `height`. Insert new rows with `Matrix::push_row`.
    pub fn new_empty(width: usize, height: usize) -> Matrix<T> {
        Matrix::new(width, height, Vec::with_capacity(height))
    }

    /// use with `Matrix::new_unchecked`.
    pub fn push_row(&mut self, row: Vec<T>) {
        assert!(self.elements.len() < self.height);
        assert_eq!(row.len(), self.width);
        self.elements.push(row);
    }
    */

    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&Vec<T>> {
        self.elements.get(y)
    }

    pub fn get(&self, y: usize, x: usize) -> Option<&T> {
        self.get_row(y).map(|row| row.get(x)).flatten()
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &Vec<T>> {
        self.elements.iter()
    }

    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Vec<T>> {
        self.elements.iter_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.iter_rows().map(|row| row.iter()).flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.iter_rows_mut().map(|row| row.iter_mut()).flatten()
    }

    /// (width, height)
    pub fn get_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

impl<T: Ring + Clone> Matrix<T> {
    pub fn with_zeros(width: usize, height: usize) -> Matrix<T> {
        Matrix::with_default(width, height, T::ZERO)
    }

    /// Creates the `n` by `n` identity Matrix.
    pub fn identity(n: usize) -> Matrix<T> {
        (0..n).into_iter().fold(Matrix::with_zeros(n, n), |mut acc, i| {
            acc.elements[i][i] = T::ONE;
            acc
        })
    }
}

impl<T: Clone> Matrix<T> {
    /// Uses first row for matrix width. All other rows are lengthed with
    /// `default` or shortend to fit dimensions.
    pub fn from_rows_or(rows: Vec<Vec<T>>, default: T) -> Matrix<T> {
        let width = rows.get(0).map(Vec::len).unwrap_or(0);
        let height = rows.len();
        Matrix {
            width,
            height,
            elements: rows.into_iter().map(|row| row.set_length(width, default.clone())).collect(),
        }
    }

    pub fn with_default(width: usize, height: usize, default: T) -> Matrix<T> {
        Matrix::new(width, height, vec![vec![default; width]; height])
    }
}

impl Matrix<f64> {
    pub fn new_random(
        width: usize,
        height: usize,
        rng: &mut DistIter<impl Distribution<f64>, RngWrapper, f64>,
    ) -> Matrix<f64> {
        Matrix {
            width,
            height,
            elements: rng.chunks(width).into_iter().take(height).map(Iterator::collect).collect(),
        }
    }
}

impl<T> std::ops::Mul<&Vec<T>> for &Matrix<T>
where T: Debug + Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>
{
    type Output = Vec<T>;

    fn mul(self, rhs: &Vec<T>) -> Self::Output {
        assert_eq!(self.width, rhs.len(), "Vector has incompatible dimensions",);
        self.elements.iter().map(|row| dot_product(&row, &rhs)).collect::<Vec<T>>()
    }
}

macro_rules! impl_mul {
    ( $type:ty : $( $rhs:ty )* ) => { $(
        impl<T> std::ops::Mul<$rhs> for $type
        where
            T: Debug + Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
        {
            type Output = Vec<T>;

            fn mul(self, rhs: $rhs) -> Self::Output {
                assert_eq!(self.width, rhs.len(), "Vector has incompatible dimensions (expected: {}, got: {})", self.width, rhs.len());
                self.elements.iter()
                    .map(|row| dot_product(&row, &rhs))
                    .collect::<Vec<T>>()
            }
        }
    )* };
}

impl_mul! { Matrix<T>: Vec<T> &Vec<T> }
impl_mul! { &Matrix<T>: Vec<T> }

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.elements[index.0][index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.elements[index.0][index.1]
    }
}

impl<T: Display> Matrix<T> {
    pub fn to_string_with_title(&self, title: impl ToString) -> Result<String, std::fmt::Error> {
        let title = title.to_string();
        let mut title = title.trim();

        let column_widths = self.elements.iter().fold(vec![0; self.width], |acc, row| {
            row.iter().zip(acc).map(|(e, max)| e.to_string().len().max(max)).collect()
        });

        let content_width = column_widths.len() + column_widths.iter().sum::<usize>() - 1;
        let content = self
            .elements
            .iter()
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

        let mut buf = String::with_capacity((content_width + 4) * (self.height + 2));

        writeln!(&mut buf, "┌ {}{}{} ┐", pre_title_pad, title, post_title_pad)?;
        writeln!(&mut buf, "{}", content)?;
        write!(&mut buf, "└ {} ┘", " ".repeat(content_width))?;
        Ok(buf)
    }
}

impl<T: Display> Display for Matrix<T> {
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
