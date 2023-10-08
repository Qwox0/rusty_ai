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

pub fn dot_product<T>(vec1: &[T], vec2: &[T]) -> T
where T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> {
    assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}
