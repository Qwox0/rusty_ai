use rusty_ai::matrix::Matrix;
use test::{black_box, Bencher};

trait MatrixBenchmarks {
    fn get_max_column_width1(&self) -> Vec<usize>;
    fn get_max_column_width2(&self) -> Vec<usize>;
    fn get_max_column_width3(&self) -> Vec<usize>;
    fn get_max_column_width4(&self) -> Vec<usize>;
    fn get_max_column_width5(&self) -> Vec<usize>;
}

impl<T> MatrixBenchmarks for Matrix<T>
where
    T: std::fmt::Display,
{
    fn get_max_column_width1(&self) -> Vec<usize> {
        let mut column_widths = vec![0; *self.get_width()];
        for row in self.get_elements().iter() {
            for (idx, element) in row.iter().enumerate() {
                column_widths[idx] = element.to_string().len().max(column_widths[idx]);
            }
        }
        column_widths
    }

    fn get_max_column_width2(&self) -> Vec<usize> {
        let mut column_widths = vec![0; *self.get_width()];
        for row in self.get_elements().iter() {
            for (max, element) in column_widths.iter_mut().zip(row) {
                *max = element.to_string().len().max(*max)
            }
        }
        column_widths
    }

    fn get_max_column_width3(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |mut acc, row| {
                for (max, element) in acc.iter_mut().zip(row) {
                    *max = element.to_string().len().max(*max)
                }
                acc
            })
    }

    fn get_max_column_width4(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |acc, row| {
                row.iter()
                    .zip(acc)
                    .map(|(e, max)| std::cmp::max(max, e.to_string().len()))
                    .collect()
            })
    }

    fn get_max_column_width5(&self) -> Vec<usize> {
        self.get_elements()
            .iter()
            .fold(vec![0; *self.get_width()], |acc, row| {
                acc.into_iter()
                    .zip(row)
                    .map(|(max, e)| std::cmp::max(max, e.to_string().len()))
                    .collect()
            })
    }
}

macro_rules! make_benches {
        ( $($name:ident: $fn:ident),* ) => { $(
            #[bench]
            fn $name(b: &mut Bencher) {
                let m = Matrix::new_random(30, 20);
                b.iter(|| {
                    black_box(Matrix::$fn(&m));
                    /*
                    for _ in 0..10 {
                        black_box(Matrix::$fn(&m));
                    }
                    */
                })
            }
        )* };
    }

make_benches! {
    bench1: get_max_column_width1,
    bench2: get_max_column_width2,
    bench3: get_max_column_width3,
    bench4: get_max_column_width4,
    bench5: get_max_column_width5
}
