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
