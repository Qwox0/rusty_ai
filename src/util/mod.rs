mod entrywise_arithmetic;
mod lerp;
mod rng;

pub use entrywise_arithmetic::*;
pub use lerp::*;
pub use rng::*;

#[allow(unused)]
pub fn cpu_count() -> usize {
    std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8)
}
