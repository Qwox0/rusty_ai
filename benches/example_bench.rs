#![feature(test)]

extern crate test;

use test::*;

#[bench]
fn bench(b: &mut Bencher) {
    b.iter(|| black_box(1 + 1))
}
