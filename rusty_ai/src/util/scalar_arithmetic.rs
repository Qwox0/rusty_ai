pub trait ScalarAdd: Sized {
    /// performs scalar addition by mutating `self` in place.
    /// `self = self + scalar`
    fn add_scalar_mut(&mut self, scalar: f64) -> &mut Self;
    /// performs scalar addition and returns result.
    /// `return self + scalar`
    fn add_scalar(mut self, scalar: f64) -> Self {
        self.add_scalar_mut(scalar);
        self
    }
}

pub trait ScalarSub: Sized {
    /// performs scalar subtraction by mutating `self` in place.
    /// `self = self - scalar`
    fn sub_scalar_mut(&mut self, scalar: f64) -> &mut Self;
    /// performs scalar subtraction and returns result.
    /// `return self - scalar`
    fn sub_scalar(mut self, scalar: f64) -> Self {
        self.sub_scalar_mut(scalar);
        self
    }
}

pub trait ScalarMul: Sized {
    /// performs scalar multiplication by mutating `self` in place.
    /// `self = self * scalar`
    fn mul_scalar_mut(&mut self, scalar: f64) -> &mut Self;
    /// performs scalar multiplication and returns result.
    /// `return self * scalar`
    fn mul_scalar(mut self, scalar: f64) -> Self {
        self.mul_scalar_mut(scalar);
        self
    }
}

pub trait ScalarDiv: Sized {
    /// performs scalar division by mutating `self` in place.
    /// `self = self / scalar`
    fn div_scalar_mut(&mut self, scalar: f64) -> &mut Self;
    /// performs scalar division and returns result.
    /// `return self / scalar`
    fn div_scalar(mut self, scalar: f64) -> Self {
        self.div_scalar_mut(scalar);
        self
    }
}

macro_rules! impl_scalar_arithmetic {
    ( $trait:ident : $trait_fn:ident $op:tt ) => {
        impl $trait for f64 {
            fn $trait_fn(&mut self, scalar: f64) -> &mut Self {
                *self $op scalar;
                self
            }
        }

        impl<T: $trait> $trait for Vec<T> {
            fn $trait_fn(&mut self, scalar: f64) -> &mut Self {
                for x in self.iter_mut() {
                    x.$trait_fn(scalar);
                }
                self
            }
        }
    };
}

impl_scalar_arithmetic! { ScalarAdd : add_scalar_mut += }
impl_scalar_arithmetic! { ScalarSub : sub_scalar_mut -= }
impl_scalar_arithmetic! { ScalarMul : mul_scalar_mut *= }
impl_scalar_arithmetic! { ScalarDiv : div_scalar_mut /= }

