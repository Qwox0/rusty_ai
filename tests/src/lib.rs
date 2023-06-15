#![feature(test)]

pub trait ScalarMul: Sized {
    fn mul_scalar_mut(&mut self, scalar: f64);
    fn mul_scalar_mut2(&mut self, scalar: f64) -> &mut Self {
        self.mul_scalar_mut(scalar);
        self
    }
    fn mul_scalar(self, scalar: f64) -> Self;
}

impl ScalarMul for f64 {
    fn mul_scalar_mut(&mut self, scalar: f64) {
        *self *= scalar;
    }

    fn mul_scalar(self, scalar: f64) -> Self {
        self * scalar
    }
}

impl<T: ScalarMul> ScalarMul for Vec<T> {
    fn mul_scalar_mut(&mut self, scalar: f64) {
        for x in self.iter_mut() {
            x.mul_scalar_mut(scalar);
        }
    }

    fn mul_scalar(self, scalar: f64) -> Self {
        self.into_iter().map(|x| x.mul_scalar(scalar)).collect()
    }
}

mod bench_mut {
    fn test_setup() -> Vec<Vec<f64>> {
        let width = 1000;
        let height = 1000;

        (0..height)
            .map(|y| (0..width).map(|x| x + y).map(f64::from).collect())
            .collect()
    }

    extern crate test;
    use super::*;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_new(b: &mut Bencher) {
        b.iter(black_box(|| {
            let a = test_setup();
            let out = black_box(a.mul_scalar(black_box(69420.0)));
        }))
    }

    #[bench]
    fn bench_mut(b: &mut Bencher) {
        b.iter(black_box(|| {
            let mut a = test_setup();
            black_box(a.mul_scalar_mut(black_box(69420.0)));
        }))
    }

    #[bench]
    fn bench_mut2(b: &mut Bencher) {
        b.iter(black_box(|| {
            let mut a = test_setup();
            black_box(a.mul_scalar_mut2(black_box(69420.0)));
        }))
    }
}
