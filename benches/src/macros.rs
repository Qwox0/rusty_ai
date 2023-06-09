macro_rules! make_benches {
    ( $type: ty; $setup: expr; $( $fn:ident $( : $arg: expr )? )* ) => { $(
        #[bench]
        fn $fn(b: &mut test::Bencher) {
            let m = $setup;
            b.iter(|| {
                for _ in 0..crate::constants::ITERATIONS {
                    test::black_box(<$type>::$fn(test::black_box(&m) $(, test::black_box($arg) )? ));
                }
            })
        }
    )* };
}
pub(crate) use make_benches;
