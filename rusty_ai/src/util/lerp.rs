pub trait Lerp<Rhs = Self>: Sized {
    /// `self = self * t + other * (1 - t)` (same as `self = t * (self - other) + other`)
    fn lerp_mut(&mut self, other: Rhs, blend: f64) -> &mut Self;

    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    fn lerp(mut self, other: Rhs, blend: f64) -> Self {
        self.lerp_mut(other, blend);
        self
    }
}

impl Lerp<f64> for f64 {
    fn lerp_mut(&mut self, other: f64, blend: f64) -> &mut Self {
        *self = blend.mul_add(*self - other, other);
        self
    }
}
