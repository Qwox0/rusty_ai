use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::{const_matrix::Matrix, util::dot_product2};

trait MatrixBenchmarks<const W: usize, const H: usize> {
    fn mul_for(&self, rhs: [f64; W]) -> [f64; H];
    fn mul_for_itermut(&self, rhs: [f64; W]) -> [f64; H];
    fn mul_map(&self, rhs: [f64; W]) -> [f64; H];
    fn mul_collect(&self, rhs: [f64; W]) -> [f64; H];
}

impl<const W: usize, const H: usize> MatrixBenchmarks<W, H> for Matrix<f64, W, H> {
    fn mul_for(&self, rhs: [f64; W]) -> [f64; H] {
        let mut res = [f64::default(); H];
        for (i, row) in self.get_elements().iter().enumerate() {
            res[i] = dot_product2(row, &rhs);
        }
        res
    }

    fn mul_for_itermut(&self, rhs: [f64; W]) -> [f64; H] {
        let mut res = [0.0; H];
        for (row, res) in self.get_elements().iter().zip(res.iter_mut()) {
            *res = dot_product2(row, &rhs);
        }
        res
    }

    fn mul_map(&self, rhs: [f64; W]) -> [f64; H] {
        self.get_elements().map(|row| dot_product2(&row, &rhs))
    }

    fn mul_collect(&self, rhs: [f64; W]) -> [f64; H] {
        self.get_elements()
            .iter()
            .map(|row| dot_product2(row, &rhs))
            .collect::<Vec<_>>()
            .try_into()
            .expect("could convert Vec<f64> to [f64; H]")
    }
}

mod benches {
    use super::*;
    make_benches! {
        Matrix<f64, MUL_SIZE_W, MUL_SIZE_H>;
        Matrix::new_random();
        mul_for(MUL_VEC)
        mul_for_itermut(MUL_VEC)
        mul_map(MUL_VEC)
        mul_collect(MUL_VEC)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    macro_rules! make_tests {
        ( $type: ty; $setup: expr; $( $fn:ident $( ( $arg: expr ) )? )* ) => { $(
            #[test]
            fn $fn() {
                let m = $setup;
                let out = <$type>::$fn(&m $(, $arg )? );
                println!("{}",m);
                $( print!("*{:?}",$arg); )?
                println!("-> {:?}", out);
                assert!(false)
            }
        )* };
    }
    make_tests! {
        Matrix<f64, MUL_SIZE_W, MUL_SIZE_H>;
        Matrix::new_random();
        mul_for(MUL_VEC)
        mul_for_itermut(MUL_VEC)
        mul_map(MUL_VEC)
        mul_collect(MUL_VEC)
    }
}
