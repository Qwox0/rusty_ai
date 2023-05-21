pub trait Lerp<Rhs = Self>: Sized {
    /// `self = self * t + other * (1 - t)` (same as `self = t * (self - other) + other`)
    fn lerp_mut(&mut self, other: Rhs, blend: f64) -> &mut Self;

    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    fn lerp(mut self, other: Rhs, blend: f64) -> Self {
        self.lerp_mut(other, blend);
        self
    }
}

impl Lerp<&f64> for f64 {
    fn lerp_mut(&mut self, other: &f64, blend: f64) -> &mut Self {
        *self = blend.mul_add(*self - *other, *other);
        self
    }
}

impl<'a, T: Lerp<&'a T> + 'a> Lerp<&'a Vec<T>> for Vec<T> {
    fn lerp_mut(&mut self, other: &'a Vec<T>, blend: f64) -> &mut Self {
        assert_eq!(self.len(), other.len());
        for (x, y) in self.iter_mut().zip(other) {
            x.lerp_mut(y, blend);
        }
        self
    }
}

impl<T: for<'a> Lerp<&'a T>> Lerp<T> for T {
    fn lerp_mut(&mut self, other: T, blend: f64) -> &mut Self {
        self.lerp_mut(&other, blend)
    }
}
