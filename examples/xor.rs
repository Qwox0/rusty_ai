#![feature(test)]
#![feature(iter_array_chunks)]

use matrix::{Float, Num};
use rand::Rng;
use rand_distr::{Bernoulli, Distribution};
use rusty_ai::{
    data::PairList,
    loss_function::{LossFunction, SquaredError},
    optimizer::{self, sgd::SGD_},
    trainer::NNTrainer,
    ActivationFn, BuildLayer, Initializer, NNBuilder, Norm,
};
use std::borrow::Borrow;

const LOSS_FUNCTION: SquaredError = SquaredError;
#[derive(Debug)]
struct XorLoss;

impl<F: Float> LossFunction<F, 1> for XorLoss {
    type ExpectedOutput = bool;

    fn propagate(&self, output: &[F; 1], expected_output: impl Borrow<Self::ExpectedOutput>) -> F {
        LOSS_FUNCTION.propagate(output, [F::from_bool(expected_output.borrow().clone())])
    }

    fn backpropagate_arr(
        &self,
        output: &[F; 1],
        expected_output: impl Borrow<Self::ExpectedOutput>,
    ) -> rusty_ai::prelude::OutputGradient<F> {
        LOSS_FUNCTION.backpropagate_arr(output, [F::from_bool(expected_output.borrow().clone())])
    }
}

fn get_nn<F: Float>(hidden_neurons: usize) -> NNTrainer<F, 2, 1, XorLoss, SGD_<F>>
where rand_distr::StandardNormal: Distribution<F> {
    NNBuilder::default()
        .double_precision()
        .element_type::<F>()
        .input::<2>()
        .layer(hidden_neurons, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::ReLU)
        .layer(1, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::Sigmoid)
        .build::<1>()
        .to_trainer()
        .loss_function(XorLoss)
        .optimizer(optimizer::sgd::SGD::default())
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build()
}

fn gen_data<X: Num>(rng: &mut impl Rng, count: usize) -> PairList<X, 2, bool> {
    Bernoulli::new(0.5)
        .unwrap()
        .sample_iter(rng)
        .array_chunks()
        .take(count)
        .map(|[in1, in2]| ([X::from_bool(in1), X::from_bool(in2)], in1 ^ in2))
        .collect()
}

fn main() {
    const HIDDEN_NEURONS: usize = 5;
    const TRAINING_DATA_COUNT: usize = 1000;
    const EPOCHS: usize = 1000;

    let mut ai = get_nn::<f64>(HIDDEN_NEURONS);

    let mut rng = rand::thread_rng();

    let mut training_data = gen_data(&mut rng, TRAINING_DATA_COUNT);
    let test_data = gen_data(&mut rng, 30);

    for epoch in 1..=EPOCHS {
        ai.train(&training_data).execute();
        training_data.shuffle_rng(&mut rng);

        if epoch % 100 == 0 {
            let loss_sum: f64 = ai.test_batch(test_data.iter()).map(|t| t.1).sum();
            println!("epoch: {:>4}, loss: {:<20}", epoch, loss_sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        const HIDDEN_NEURONS: usize = 5;
        const TRAINING_DATA_COUNT: usize = 500;
        const EPOCHS: usize = 500;

        let mut ai = get_nn::<f32>(HIDDEN_NEURONS);

        let mut rng = rand::thread_rng();
        let mut training_data = gen_data(&mut rng, TRAINING_DATA_COUNT);

        for _ in 1..=EPOCHS {
            ai.train(&training_data).execute();
            training_data.shuffle_rng(&mut rng);
        }

        for (input, eo) in gen_data(&mut rng, 30) {
            let (_, loss) = ai.test(&input, &eo);
            assert!(
                loss < 1e-5,
                "assertion failed: loss < 1e-8 (loss: {}). This test contains rng. Please repeat \
                 the test to ensure that it has failed.",
                loss
            )
        }
    }
}

#[macro_export]
macro_rules! _bench_example_epoch {
    ( $( $bench_name:ident : $neurons:expr, $data_count:expr );* $(;)? ) => { $(
        #[bench]
        fn $bench_name(b: &mut Bencher) {
            let mut ai = get_nn::<f32>($neurons);
            let mut rng = rand::thread_rng();
            let mut training_data = gen_data(&mut rng, $data_count);
            b.iter(|| {
                black_box(Training::execute(black_box(NNTrainer::train(
                    black_box(&mut ai),
                    black_box(&training_data),
                ))));
                training_data.shuffle_rng(&mut rng);
            })
        }
    )* };
}
pub use _bench_example_epoch as bench_example_epoch;

/// # Results (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_003 ... bench:      31,245 ns/iter (+/- 4,383)
/// test benches::epoch_datacount_100_neurons_005 ... bench:      30,120 ns/iter (+/- 8,429)
/// test benches::epoch_datacount_100_neurons_010 ... bench:      34,837 ns/iter (+/- 3,500)
/// test benches::epoch_datacount_100_neurons_100 ... bench:     109,016 ns/iter (+/- 28,412)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     431,355 ns/iter (+/- 110,238)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     758,352 ns/iter (+/- 221,523)
/// test benches::epoch_datacount_500_neurons_003 ... bench:     147,573 ns/iter (+/- 30,456)
/// test benches::epoch_datacount_500_neurons_005 ... bench:     140,617 ns/iter (+/- 17,428)
/// test benches::epoch_datacount_500_neurons_010 ... bench:     183,356 ns/iter (+/- 23,351)
/// test benches::epoch_datacount_500_neurons_030 ... bench:     267,253 ns/iter (+/- 30,388)
/// test benches::epoch_datacount_500_neurons_050 ... bench:     349,008 ns/iter (+/- 35,042)
/// test benches::epoch_datacount_500_neurons_070 ... bench:     435,233 ns/iter (+/- 76,669)
/// test benches::epoch_datacount_500_neurons_090 ... bench:     490,954 ns/iter (+/- 66,953)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     601,599 ns/iter (+/- 170,455)
/// test benches::epoch_datacount_500_neurons_200 ... bench:     978,337 ns/iter (+/- 1,711,630)
/// test benches::epoch_datacount_500_neurons_300 ... bench:   1,345,359 ns/iter (+/- 123,731)
/// test benches::epoch_datacount_500_neurons_400 ... bench:   1,762,124 ns/iter (+/- 333,995)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   2,108,879 ns/iter (+/- 351,741)
/// test benches::epoch_datacount_500_neurons_600 ... bench:   2,654,988 ns/iter (+/- 494,224)
/// test benches::epoch_datacount_500_neurons_700 ... bench:   2,861,289 ns/iter (+/- 429,318)
/// test benches::epoch_datacount_500_neurons_800 ... bench:   3,406,868 ns/iter (+/- 870,594)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   3,705,758 ns/iter (+/- 653,876)
/// test benches::epoch_datacount_900_neurons_003 ... bench:     251,537 ns/iter (+/- 67,355)
/// test benches::epoch_datacount_900_neurons_005 ... bench:     244,351 ns/iter (+/- 14,767)
/// test benches::epoch_datacount_900_neurons_010 ... bench:     297,827 ns/iter (+/- 25,308)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     975,830 ns/iter (+/- 313,410)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   3,930,385 ns/iter (+/- 587,206)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   6,668,852 ns/iter (+/- 985,858)
/// ```
///
/// # Results with rayon (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_003 ... bench:      44,367 ns/iter (+/- 3,261)
/// test benches::epoch_datacount_100_neurons_005 ... bench:      47,409 ns/iter (+/- 7,531)
/// test benches::epoch_datacount_100_neurons_010 ... bench:      58,925 ns/iter (+/- 4,591)
/// test benches::epoch_datacount_100_neurons_100 ... bench:     161,681 ns/iter (+/- 34,975)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     424,835 ns/iter (+/- 86,984)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     800,583 ns/iter (+/- 131,383)
/// test benches::epoch_datacount_500_neurons_003 ... bench:     143,521 ns/iter (+/- 19,674)
/// test benches::epoch_datacount_500_neurons_005 ... bench:     144,590 ns/iter (+/- 22,819)
/// test benches::epoch_datacount_500_neurons_010 ... bench:     166,325 ns/iter (+/- 15,133)
/// test benches::epoch_datacount_500_neurons_030 ... bench:     237,854 ns/iter (+/- 39,518)
/// test benches::epoch_datacount_500_neurons_050 ... bench:     303,969 ns/iter (+/- 58,847)
/// test benches::epoch_datacount_500_neurons_070 ... bench:     423,059 ns/iter (+/- 121,634)
/// test benches::epoch_datacount_500_neurons_090 ... bench:     452,006 ns/iter (+/- 140,686)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     536,609 ns/iter (+/- 198,314)
/// test benches::epoch_datacount_500_neurons_200 ... bench:     920,333 ns/iter (+/- 349,582)
/// test benches::epoch_datacount_500_neurons_300 ... bench:   1,354,631 ns/iter (+/- 610,535)
/// test benches::epoch_datacount_500_neurons_400 ... bench:   1,362,134 ns/iter (+/- 414,454)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   1,371,858 ns/iter (+/- 227,611)
/// test benches::epoch_datacount_500_neurons_600 ... bench:   1,624,216 ns/iter (+/- 283,872)
/// test benches::epoch_datacount_500_neurons_700 ... bench:   1,945,982 ns/iter (+/- 438,969)
/// test benches::epoch_datacount_500_neurons_800 ... bench:   2,148,179 ns/iter (+/- 314,916)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   2,401,087 ns/iter (+/- 414,738)
/// test benches::epoch_datacount_900_neurons_003 ... bench:     231,682 ns/iter (+/- 60,844)
/// test benches::epoch_datacount_900_neurons_005 ... bench:     237,438 ns/iter (+/- 62,057)
/// test benches::epoch_datacount_900_neurons_010 ... bench:     242,021 ns/iter (+/- 49,833)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     644,834 ns/iter (+/- 145,178)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   2,303,908 ns/iter (+/- 543,419)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   3,728,160 ns/iter (+/- 476,066)
/// ```
///
/// # Rayon (better grad init) (f64)
///
/// ```
/// test benches::epoch_datacount_500_neurons_700 ... bench:   1,582,812 ns/iter (+/- 203,955)
/// test benches::epoch_datacount_500_neurons_800 ... bench:   1,867,005 ns/iter (+/- 529,016)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   2,055,371 ns/iter (+/- 236,685)
/// test benches::epoch_datacount_900_neurons_003 ... bench:     174,620 ns/iter (+/- 19,028)
/// test benches::epoch_datacount_900_neurons_005 ... bench:     177,043 ns/iter (+/- 25,889)
/// test benches::epoch_datacount_900_neurons_010 ... bench:     200,987 ns/iter (+/- 27,207)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     555,309 ns/iter (+/- 138,537)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   1,781,902 ns/iter (+/- 577,616)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   3,185,829 ns/iter (+/- 1,289,502)
/// ```
///
/// # Rayon (better grad init) (f32)
///
/// ```
/// test benches::epoch_datacount_500_neurons_700 ... bench:   1,240,364 ns/iter (+/- 197,929)
/// test benches::epoch_datacount_500_neurons_800 ... bench:   1,374,512 ns/iter (+/- 140,880)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   1,533,832 ns/iter (+/- 132,534)
/// test benches::epoch_datacount_900_neurons_003 ... bench:     170,590 ns/iter (+/- 19,448)
/// test benches::epoch_datacount_900_neurons_005 ... bench:     179,512 ns/iter (+/- 17,479)
/// test benches::epoch_datacount_900_neurons_010 ... bench:     186,440 ns/iter (+/- 21,873)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     438,279 ns/iter (+/- 108,114)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   1,401,092 ns/iter (+/- 149,090)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   2,356,569 ns/iter (+/- 414,852)
/// ```
#[cfg(test)]
mod benches {
    use super::*;

    extern crate test;
    use rusty_ai::training::Training;
    use test::*;

    bench_example_epoch! {
        /*
        epoch_datacount_100_neurons_003: 3, 100;
        epoch_datacount_100_neurons_005: 5, 100;
        epoch_datacount_100_neurons_010: 10, 100;
        epoch_datacount_100_neurons_100: 100, 100;
        epoch_datacount_100_neurons_500: 500, 100;
        epoch_datacount_100_neurons_900: 900, 100;

        epoch_datacount_500_neurons_003: 3, 500;
        epoch_datacount_500_neurons_005: 5, 500;
        epoch_datacount_500_neurons_010: 10, 500;
        epoch_datacount_500_neurons_030: 30, 500;
        epoch_datacount_500_neurons_050: 50, 500;
        epoch_datacount_500_neurons_070: 70, 500;
        epoch_datacount_500_neurons_090: 90, 500;
        epoch_datacount_500_neurons_100: 100, 500;
        epoch_datacount_500_neurons_200: 200, 500;
        epoch_datacount_500_neurons_300: 300, 500;
        epoch_datacount_500_neurons_400: 400, 500;
        epoch_datacount_500_neurons_500: 500, 500;
        epoch_datacount_500_neurons_600: 600, 500;
        */
        epoch_datacount_500_neurons_700: 700, 500;
        epoch_datacount_500_neurons_800: 800, 500;
        epoch_datacount_500_neurons_900: 900, 500;

        epoch_datacount_900_neurons_003: 3, 900;
        epoch_datacount_900_neurons_005: 5, 900;
        epoch_datacount_900_neurons_010: 10, 900;
        epoch_datacount_900_neurons_100: 100, 900;
        epoch_datacount_900_neurons_500: 500, 900;
        epoch_datacount_900_neurons_900: 900, 900;
    }
}
