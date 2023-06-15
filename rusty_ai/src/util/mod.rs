mod entrywise_arithmetic;
mod lerp;
mod macros;
mod norm;
mod rng;
mod scalar_arithmetic;

pub use entrywise_arithmetic::*;
pub use lerp::*;
pub(crate) use macros::*;
pub use norm::*;
pub use rng::*;
pub use scalar_arithmetic::*;

pub fn dot_product<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}

pub trait SetLength {
    type Item;
    fn set_length(self, new_length: usize, default: Self::Item) -> Self;
    fn to_arr<const N: usize>(self, default: Self::Item) -> [Self::Item; N];
}

impl<T: Clone> SetLength for Vec<T> {
    type Item = T;
    fn set_length(mut self, new_length: usize, default: Self::Item) -> Self {
        self.resize(new_length, default);
        self
    }

    fn to_arr<const N: usize>(self, default: Self::Item) -> [Self::Item; N] {
        let mut arr = std::array::from_fn::<_, N, _>(|_| default.clone());
        for (idx, elem) in self.into_iter().enumerate() {
            arr[idx] = elem;
        }
        arr
    }
}

pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(8)
}
