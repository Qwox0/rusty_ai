#![feature(test)]

//! # Results
//!
//! ```
//! test alloc_id_f32_0004 ... bench:          86 ns/iter (+/- 5)
//! test alloc_id_f32_0032 ... bench:       1,012 ns/iter (+/- 25)
//! test alloc_id_f32_0256 ... bench:      38,297 ns/iter (+/- 2,369)
//! test alloc_id_f32_0512 ... bench:     241,854 ns/iter (+/- 8,775)
//! test alloc_id_f32_1024 ... bench:   1,081,313 ns/iter (+/- 289,568)
//! test alloc_id_f64_0004 ... bench:          74 ns/iter (+/- 1)
//! test alloc_id_f64_0032 ... bench:       1,056 ns/iter (+/- 239)
//! test alloc_id_f64_0256 ... bench:      94,741 ns/iter (+/- 2,351)
//! test alloc_id_f64_0512 ... bench:     440,596 ns/iter (+/- 23,326)
//! test alloc_id_f64_1024 ... bench:   2,294,647 ns/iter (+/- 124,985)
//! test mul_vec_0004      ... bench:          24 ns/iter (+/- 1)
//! test mul_vec_0032      ... bench:         440 ns/iter (+/- 27)
//! test mul_vec_0256      ... bench:      44,660 ns/iter (+/- 678)
//! test mul_vec_0512      ... bench:     191,072 ns/iter (+/- 1,862)
//! test mul_vec_1024      ... bench:     803,260 ns/iter (+/- 16,343)
//! ```

#[cfg(__never_compiled)]
use crate as _docs;

extern crate test;

use matrix::Matrix;
use rand::{distributions::Uniform, Rng};
use test::*;

macro_rules! bench_mul_vec {
    ( $( $bench_name:ident : $dim:expr ),* $(,)? ) => { $(
        #[bench]
        fn $bench_name(b: &mut Bencher) {
            let mut rng_iter = rand::thread_rng().sample_iter(Uniform::new(0.0, 1.0));
            let matrix: Matrix<_> = Matrix::from_iter($dim, $dim, &mut rng_iter);
            let vec = rng_iter.take($dim).collect::<Vec<_>>();
            b.iter(|| black_box(Matrix::mul_vec(black_box(&matrix), black_box(&vec))))
        }
    )* };
}

bench_mul_vec! {
    mul_vec_0004: 4,
    //mul_vec_0008: 8,
    //mul_vec_0016: 16,
    mul_vec_0032: 32,
    //mul_vec_0064: 64,
    //mul_vec_0128: 128,
    mul_vec_0256: 256,
    mul_vec_0512: 512,
    mul_vec_1024: 1024,
}

macro_rules! bench_alloc_identity {
    ( $( $bench_name:ident : $ty:ty => $dim:expr ),* $(,)? ) => { $(
        #[bench]
        fn $bench_name(b: &mut Bencher) {
            b.iter(|| black_box(Matrix::<$ty>::identity(black_box($dim))))
        }
    )* };
}

bench_alloc_identity! {
    alloc_id_f32_0004: f32 => 4,
    alloc_id_f32_0032: f32 => 32,
    alloc_id_f32_0256: f32 => 256,
    alloc_id_f32_0512: f32 => 512,
    alloc_id_f32_1024: f32 => 1024,

    alloc_id_f64_0004: f64 => 4,
    alloc_id_f64_0032: f64 => 32,
    alloc_id_f64_0256: f64 => 256,
    alloc_id_f64_0512: f64 => 512,
    alloc_id_f64_1024: f64 => 1024,

    // similar results to floats
    // alloc_id_i32_0004: i32 => 4,
    // alloc_id_i32_0032: i32 => 32,
    // alloc_id_i32_0256: i32 => 256,
    // alloc_id_i32_0512: i32 => 512,
    // alloc_id_i32_1024: i32 => 1024,

    // alloc_id_u64_0004: u64 => 4,
    // alloc_id_u64_0032: u64 => 32,
    // alloc_id_u64_0256: u64 => 256,
    // alloc_id_u64_0512: u64 => 512,
    // alloc_id_u64_1024: u64 => 1024,
}

/// similar results to [`bench_alloc_identity`]
#[allow(unused)]
macro_rules! bench_alloc_zeros {
    ( $( $bench_name:ident : $ty:ty => $dim:expr ),* $(,)? ) => { $(
        #[bench]
        fn $bench_name(b: &mut Bencher) {
            b.iter(|| black_box(Matrix::<$ty>::with_zeros(black_box($dim), black_box($dim))))
        }
    )* };
}

// bench_alloc_zeros! {
//     alloc_zeros_f32_0004: f32 => 4,
//     alloc_zeros_f32_0032: f32 => 32,
//     alloc_zeros_f32_0256: f32 => 256,
//     alloc_zeros_f32_0512: f32 => 512,
//     alloc_zeros_f32_1024: f32 => 1024,
//
//     alloc_zeros_f64_0004: f64 => 4,
//     alloc_zeros_f64_0032: f64 => 32,
//     alloc_zeros_f64_0256: f64 => 256,
//     alloc_zeros_f64_0512: f64 => 512,
//     alloc_zeros_f64_1024: f64 => 1024,
// }
