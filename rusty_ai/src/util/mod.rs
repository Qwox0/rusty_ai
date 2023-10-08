mod assign_each;
mod entrywise_arithmetic;
mod lerp;
mod macros;
mod norm;
mod rng;

pub use assign_each::*;
pub use entrywise_arithmetic::*;
pub use lerp::*;
pub(crate) use macros::*;
pub use norm::*;
pub use rng::*;

pub fn cpu_count() -> usize {
    std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8)
}
