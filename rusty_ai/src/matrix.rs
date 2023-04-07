use crate::util::macros::impl_getter;
use crate::util::{dot_product, SetLength};
use rand::Rng;
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul};

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

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    width: usize,
    height: usize,
    elements: Vec<Vec<T>>,
}

impl_getter! { Matrix<T>:
    get_width -> width: usize,
    get_height -> height: usize,
    get_elements -> elements: Vec<Vec<T>>,
}

impl<T: Ring + Clone> Matrix<T> {
    pub fn with_zero(width: usize, height: usize) -> Matrix<T> {
        Matrix {
            width,
            height,
            elements: vec![vec![T::ZERO; width]; height],
        }
    }

    /// Creates the `n` by `n` identity Matrix.
    pub fn identity(n: usize) -> Matrix<T> {
        (0..n)
            .into_iter()
            .fold(Matrix::with_zero(n, n), |mut acc, i| {
                acc.elements[i][i] = T::ONE;
                acc
            })
    }
}

impl<T> Matrix<T> {
    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&Vec<T>> {
        self.elements.get(y)
    }

    pub fn get(&self, y: usize, x: usize) -> Option<&T> {
        self.get_row(y).map(|row| row.get(x)).flatten()
    }
}

impl<T: Clone> Matrix<T> {
    /// Uses first row for matrix width. All other rows are lengthed with `default` or shortend to
    /// fit dimensions.
    pub fn from_rows(rows: Vec<Vec<T>>, default: T) -> Matrix<T> {
        let width = rows.get(0).map(Vec::len).unwrap_or(0);
        let height = rows.len();
        Matrix {
            width,
            height,
            elements: rows
                .into_iter()
                .map(|row| row.set_length(width, default.clone()))
                .collect(),
        }
    }

    pub fn with_default(width: usize, height: usize, default: T) -> Matrix<T> {
        Matrix {
            width,
            height,
            elements: vec![vec![default; width]; height],
        }
    }
}

impl<T: Clone + Default> Matrix<T> {
    pub fn from_rows_default(rows: Vec<Vec<T>>) -> Matrix<T> {
        Matrix::from_rows(rows, T::default())
    }

    pub fn new_default(width: usize, height: usize) -> Matrix<T> {
        Matrix::with_default(width, height, T::default())
    }
}

impl Matrix<f64> {
    pub fn new_random(width: usize, height: usize) -> Matrix<f64> {
        let mut rng = rand::thread_rng();
        Matrix {
            width,
            height,
            elements: (0..height)
                .map(|_| (0..width).map(|_| rng.gen()).collect())
                .collect(),
        }
    }
}

macro_rules! impl_mul {
    ( $( $type:ty )*) => {$(
        impl<T> std::ops::Mul<Vec<T>> for $type
        where
            T: Debug + Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
        {
            type Output = Vec<T>;

            fn mul(self, rhs: Vec<T>) -> Self::Output {
                debug_assert_eq!(self.width, rhs.len(), "Vector has incompatible dimensions (expected: {}, got: {})", self.width, rhs.len());
                self.elements.iter()
                    .map(|row| dot_product(&row, &rhs))
                    .collect::<Vec<T>>()
            }
        }

    )*};
}

impl_mul! { Matrix<T> &Matrix<T> }

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let column_widths = self.elements.iter().fold(vec![0; self.width], |acc, row| {
            row.iter()
                .zip(acc)
                .map(|(e, max)| std::cmp::max(max, e.to_string().len()))
                .collect()
        });

        let full_padding =
            " ".repeat(column_widths.len() + column_widths.iter().sum::<usize>() + 1);
        write!(
            f,
            "┌{0}┐\n{1}\n└{0}┘",
            full_padding,
            self.elements
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
                .join("\n"),
        )
    }
}

#[cfg(test)]
mod tests {
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
}
