use crate::util::dot_product2;
use rand::Rng;
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T, const WIDTH: usize, const HEIGHT: usize>([[T; WIDTH]; HEIGHT]);

impl<const N: usize> Matrix<f64, N, N> {
    pub fn identity() -> Matrix<f64, N, N> {
        Matrix((0..N).into_iter().fold([[0.0; N]; N], |mut acc, i| {
            acc[i][i] = 1.0;
            acc
        }))
    }
}

impl<T: Clone, const W: usize, const H: usize> Matrix<T, W, H> {
    pub fn from_rows(rows: [[T; W]; H]) -> Matrix<T, W, H> {
        Matrix(rows)
    }
}

impl<T: Copy, const W: usize, const H: usize> Matrix<T, W, H> {
    pub fn with_default(default: T) -> Matrix<T, W, H> {
        Matrix([[default; W]; H])
    }
}

impl<const W: usize, const H: usize> Matrix<f64, W, H> {
    pub fn new_random() -> Matrix<f64, W, H> {
        let mut rng = rand::thread_rng();
        Matrix([[0; W]; H].map(|row| row.map(|_| rng.gen())))
    }
}

impl<T, const W: usize, const H: usize> Matrix<T, W, H> {
    #[inline]
    pub fn get_elements(&self) -> &[[T; W]; H] {
        &self.0
    }

    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&[T; W]> {
        self.0.get(y)
    }

    pub fn get(&self, y: usize, x: usize) -> Option<&T> {
        self.get_row(y)?.get(x)
    }
}

impl<T, const W: usize, const H: usize> std::ops::Mul<[T; W]> for &Matrix<T, W, H>
where
    T: Default + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    type Output = [T; H];

    fn mul(self, rhs: [T; W]) -> Self::Output {
        let mut res = [T::default(); H];
        for (i, row) in self.0.iter().enumerate() {
            res[i] = dot_product2(row, &rhs);
        }
        res
        /*
        self.0.map(|row| dot_product2(&row, &rhs))

        self.0
            .iter()
            .map(|row| dot_product2(row, &rhs))
            .collect::<Vec<_>>()
            .try_into()
            .expect("could convert Vec<f64> to [f64; H]")
            */
    }
}

impl<T: Display, const W: usize, const H: usize> std::fmt::Display for Matrix<T, W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let column_widths = self.0.iter().fold(vec![0; W], |acc, row| {
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
            self.0
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
