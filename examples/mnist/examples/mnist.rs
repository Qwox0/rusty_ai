#![feature(test)]

fn main() {
    mnist::main();
}

#[cfg(test)]
mod tests_ {
    extern crate test;

    use test::*;

    /// # Hack
    /// test tests_::test_propagate  ... bench:      83,896 ns/iter (+/- 3,487)
    /// test tests_::test_propagate  ... bench:      85,949 ns/iter (+/- 3,861)
    /// test tests_::test_propagate  ... bench:      87,303 ns/iter (+/- 3,654)
    /// test tests_::test_propagate  ... bench:      87,310 ns/iter (+/- 3,533)
    /// test tests_::test_propagate  ... bench:      87,229 ns/iter (+/- 3,064) median
    #[bench]
    fn test_propagate(b: &mut Bencher) {
        mnist::tests::test_propagate(b);
    }

    /// # Hack
    /// test tests_::test_train_hack ... bench:     730,855 ns/iter (+/- 238,087) median
    /// test tests_::test_train_hack ... bench:     637,176 ns/iter (+/- 67,747)
    /// test tests_::test_train_hack ... bench:     683,097 ns/iter (+/- 214,770)
    /// test tests_::test_train_hack ... bench:     746,663 ns/iter (+/- 235,444)
    /// test tests_::test_train_hack ... bench:     854,018 ns/iter (+/- 244,765)
    #[bench]
    fn test_train_hack(b: &mut Bencher) {
        mnist::tests::test_train_hack(b);
    }
}
