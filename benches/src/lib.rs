#![feature(test)]
extern crate test;

pub(crate) mod macros;


#[cfg(test)]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[bench]
fn test_bench(b: &mut test::Bencher) {
    use test::black_box;

    b.iter(|| {
        for _ in 0..crate::constants::ITERATIONS {
            black_box(add(black_box(1), black_box(2)));
        }
    })
}
