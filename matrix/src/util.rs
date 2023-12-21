pub trait SetLength {
    type Item;
    fn set_length(self, new_length: usize, default: Self::Item) -> Self;
}

impl<T: Clone> SetLength for Vec<T> {
    type Item = T;

    fn set_length(mut self, new_length: usize, default: Self::Item) -> Self {
        self.resize(new_length, default);
        self
    }
}

/// Calculated the dot product of the two vectors.
pub fn dot_product<T>(vec1: &[T], vec2: &[T]) -> T
where T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> {
    assert_eq!(vec1.len(), vec2.len());
    dot_product_unchecked(vec1, vec2)
}

/// Calculated the dot product of the two vectors.
///
/// This doesn't perform the length equality check. If the lengths of the input vectors aren't
/// equal, no output value is guaranteed.
pub fn dot_product_unchecked<T>(vec1: &[T], vec2: &[T]) -> T
where T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> {
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}
