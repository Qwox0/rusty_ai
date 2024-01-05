/*
use matrix::Float;

pub trait Lerp<Rhs = Self>: Sized {
    /// `self = self * t + other * (1 - t)` (same as `self = t * (self - other) + other`)
    fn lerp_mut(&mut self, other: Rhs, blend: Self) -> &mut Self;

    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    fn lerp(mut self, other: Rhs, blend: Self) -> Self {
        self.lerp_mut(other, blend);
        self
    }
}

impl<F: Float> Lerp<F> for F {
    fn lerp_mut(&mut self, other: F, blend: F) -> &mut Self {
        *self = blend.mul_add(*self - other, other);
        self
    }
}
*/
