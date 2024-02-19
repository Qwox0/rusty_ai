#![allow(unused)]

/// Returns the available cpu count. see [`std::thread::available_parallelism`].
///
/// The default value is `8`.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8)
}
