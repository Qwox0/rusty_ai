use crate::util::macros::impl_getter;
use crate::util::{dot_product, dot_product2, SetLength};
use rand::Rng;
use std::fmt::{Debug, Display};

#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    width: usize,
    height: usize,
    elements: Vec<Vec<T>>,
}

impl_getter! { Matrix<T>:
    get_width -> width: usize,
    get_height -> height: usize,
    get_elements -> elements: Vec<Vec<T>>
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

/*
pub struct Matrix<T, const WIDTH: usize, const HEIGHT: usize> {
    elements: [[T; WIDTH]; HEIGHT],
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Matrix<T, WIDTH, HEIGHT>
where
    T: Copy,
{
    pub fn with_default(default: T) -> Matrix<T, WIDTH, HEIGHT> {
        Matrix {
            elements: [[default; WIDTH]; HEIGHT],
        }
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Matrix<T, WIDTH, HEIGHT> {
    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&[T; WIDTH]> {
        self.elements.get(y)
    }

    pub fn get(&self, y: usize, x: usize) -> Option<&T> {
        self.get_row(y).map(|row| row.get(x)).flatten()
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> std::ops::Mul<[T; WIDTH]>
    for Matrix<T, WIDTH, HEIGHT>
where
    T: Debug + Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    type Output = [T; WIDTH];

    fn mul(self, rhs: [T; WIDTH]) -> Self::Output {
        rhs.iter()
            .enumerate()
            .map(|(idx, _)| dot_product(&self.get_row(idx).expect(""), &rhs))
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }
}
*/

impl<T> std::fmt::Display for Matrix<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let column_widths = self.elements.iter().fold(vec![0; self.width], |acc, row| {
            row.iter()
                .zip(acc)
                .map(|(e, max)| std::cmp::max(max, e.to_string().len()))
                .collect()
        });

        // println!("{:?}", column_widths);

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

impl<T> std::fmt::Debug for Matrix<T>
where
    //T: std::fmt::Display,
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matrix")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("elements", &self.elements)
            .finish()
    }
}

