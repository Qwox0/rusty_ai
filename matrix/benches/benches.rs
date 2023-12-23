#![feature(test)]

//! # Results without optimization
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
//!
//! # Results with `Box<[T]>`
//!
//! ```
//! test alloc_id_f32_0004 ... bench:          44 ns/iter (+/- 39)
//! test alloc_id_f32_0032 ... bench:         138 ns/iter (+/- 24)
//! test alloc_id_f32_0256 ... bench:       4,749 ns/iter (+/- 1,847)
//! test alloc_id_f32_0512 ... bench:      26,577 ns/iter (+/- 1,825)
//! test alloc_id_f32_1024 ... bench:     103,882 ns/iter (+/- 9,373)
//! test alloc_id_f64_0004 ... bench:          41 ns/iter (+/- 4)
//! test alloc_id_f64_0032 ... bench:         201 ns/iter (+/- 37)
//! test alloc_id_f64_0256 ... bench:      12,970 ns/iter (+/- 602)
//! test alloc_id_f64_0512 ... bench:      37,747 ns/iter (+/- 5,489)
//! test alloc_id_f64_1024 ... bench:     604,506 ns/iter (+/- 445,550)
//! test mul_vec_0004      ... bench:          23 ns/iter (+/- 5)
//! test mul_vec_0032      ... bench:         425 ns/iter (+/- 90)
//! test mul_vec_0256      ... bench:      50,055 ns/iter (+/- 4,626)
//! test mul_vec_0512      ... bench:     204,197 ns/iter (+/- 6,917)
//! test mul_vec_1024      ... bench:     868,463 ns/iter (+/- 37,307)
//! ```
//!
//! # Results with `Box<[T]>` and own [`matrix::iter_rows::IterRows`]
//!
//! ```
//! test alloc_id_f32_0004 ... bench:          20 ns/iter (+/- 0)
//! test alloc_id_f32_0032 ... bench:         106 ns/iter (+/- 3)
//! test alloc_id_f32_0256 ... bench:       4,502 ns/iter (+/- 108)
//! test alloc_id_f32_0512 ... bench:      18,092 ns/iter (+/- 7,139)
//! test alloc_id_f32_1024 ... bench:      75,477 ns/iter (+/- 13,815)
//! test alloc_id_f64_0004 ... bench:          38 ns/iter (+/- 3)
//! test alloc_id_f64_0032 ... bench:         198 ns/iter (+/- 5)
//! test alloc_id_f64_0256 ... bench:       9,028 ns/iter (+/- 65)
//! test alloc_id_f64_0512 ... bench:      49,760 ns/iter (+/- 14,681)
//! test alloc_id_f64_1024 ... bench:     425,785 ns/iter (+/- 111,819)
//! test from_arr_f32_0004 ... bench:          21 ns/iter (+/- 0)
//! test from_arr_f32_0032 ... bench:         333 ns/iter (+/- 23)
//! test from_arr_f32_0256 ... bench:      36,119 ns/iter (+/- 9,583)
//! test from_arr_f32_0512 ... bench:     153,749 ns/iter (+/- 4,601)
//! test from_arr_f64_0004 ... bench:          33 ns/iter (+/- 1)
//! test from_arr_f64_0032 ... bench:         617 ns/iter (+/- 108)
//! test from_arr_f64_0256 ... bench:      76,034 ns/iter (+/- 18,698)
//! test mul_vec_0004      ... bench:          28 ns/iter (+/- 0)
//! test mul_vec_0032      ... bench:         446 ns/iter (+/- 6)
//! test mul_vec_0256      ... bench:      44,688 ns/iter (+/- 4,253)
//! test mul_vec_0512      ... bench:     190,551 ns/iter (+/- 13,061)
//! test mul_vec_1024      ... bench:     803,469 ns/iter (+/- 20,467)
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

macro_rules! bench_from_arr {
    ( $( $bench_name:ident : $ty:ty => $dim:expr ),* $(,)? ) => { $(
        #[bench]
        fn $bench_name(b: &mut Bencher) {
            b.iter(|| black_box(Matrix::<$ty>::from(black_box([[<$ty>::default(); $dim]; $dim]))))
        }
    )* };
}

bench_from_arr! {
    from_arr_f32_0004: f32 => 4,
    from_arr_f32_0032: f32 => 32,
    from_arr_f32_0256: f32 => 256,
    from_arr_f32_0512: f32 => 512,
    // from_arr_f32_1024: f32 => 1024, // -> Stack overflow

    from_arr_f64_0004: f64 => 4,
    from_arr_f64_0032: f64 => 32,
    from_arr_f64_0256: f64 => 256,
    // from_arr_f64_0512: f64 => 512, // -> Stack overflow
    // from_arr_f64_1024: f64 => 1024, // -> Stack overflow
}
