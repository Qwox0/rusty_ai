#![feature(test)]
#![feature(iter_array_chunks)]

use const_tensor::{vector, Float, Num, Tensor, Vector};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};
use rusty_ai::{
    initializer::PytorchDefault,
    loss_function::{LossFunction, SquaredError},
    nn::Pair,
    optimizer::{self, sgd::SGD},
    trainer::NNTrainer,
    NNBuilder, Norm, NN,
};

const LOSS_FUNCTION: SquaredError = SquaredError;
#[derive(Debug)]
struct XorLoss;

impl<X: Float> LossFunction<X, [(); 1]> for XorLoss {
    type ExpectedOutput = bool;

    fn propagate(&self, output: &vector<X, 1>, expected_output: &Self::ExpectedOutput) -> X {
        let expected_output = Vector::new([X::from_bool(*expected_output)]);
        LOSS_FUNCTION.propagate(output, &expected_output)
    }

    fn backpropagate(
        &self,
        output: &vector<X, 1>,
        expected_output: &Self::ExpectedOutput,
    ) -> Vector<X, 1> {
        let expected_output = Vector::new([X::from_bool(*expected_output)]);

        LOSS_FUNCTION.backpropagate(output, &expected_output)
    }
}

fn get_nn<X: Float, const NEURONS: usize>()
-> NNTrainer<X, [(); 2], [(); 1], XorLoss, SGD<X>, impl NN<X, [(); 2], [(); 1]>>
where rand_distr::StandardNormal: Distribution<X> {
    NNBuilder::default()
        .element_type::<X>()
        //.default_rng()
        .seeded_rng(3)
        .input_shape::<[(); 2]>()
        .layer::<NEURONS>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<1>(PytorchDefault, PytorchDefault)
        .relu()
        .build()
        .to_trainer()
        .loss_function(XorLoss)
        .optimizer(optimizer::sgd::SGD::default())
        .retain_gradient(true)
        .new_clip_gradient_norm(X::lit(5), Norm::Two)
        .build()
}
/*
   NNBuilder::default()
       .seeded_rng(3)
       .input::<2>()
       .layer(hidden_neurons, PytorchDefault, PytorchDefault)
       .activation_function(ActivationFn::ReLU)
       .layer(1, PytorchDefault, PytorchDefault)
       .activation_function(ActivationFn::Sigmoid)
       .build::<1>()
       .to_trainer()
       .loss_function(XorLoss)
       .optimizer(optimizer::sgd::SGD::default())
       .retain_gradient(true)
       .new_clip_gradient_norm(5.0, Norm::Two)
       .build()
*/

fn gen_data<'a, X: Num>(
    rng: &'a mut impl Rng,
    count: usize,
) -> impl Iterator<Item = Pair<X, [(); 2], bool>> + 'a {
    Bernoulli::new(0.5)
        .unwrap()
        .sample_iter(rng)
        .array_chunks()
        .take(count)
        .map(|[in1, in2]| Pair::new(Tensor::new([in1, in2]).map_clone(X::from_bool), in1 ^ in2))
}

fn main() {
    const HIDDEN_NEURONS: usize = 5;
    const TRAINING_DATA_COUNT: usize = 1000;
    const EPOCHS: usize = 1000;

    let mut ai = get_nn::<f64, HIDDEN_NEURONS>();

    //let mut rng = rand::thread_rng();
    let mut rng = StdRng::seed_from_u64(69420);

    let mut training_data = gen_data(&mut rng, TRAINING_DATA_COUNT).collect::<Vec<_>>();
    let test_data = gen_data(&mut rng, 30).collect::<Vec<_>>();

    let loss_sum: f64 = ai.test_batch(test_data.iter()).map(|r| r.get_loss()).sum();
    println!("epoch: {:>4}, loss: {:<20}", 0, loss_sum);

    for epoch in 1..=EPOCHS {
        ai.train_single_thread(&training_data);
        //ai.train_rayon(&training_data);
        training_data.shuffle(&mut rng);

        if epoch % 100 == 0 {
            let loss_sum: f64 = ai.test_batch(test_data.iter()).map(|r| r.get_loss()).sum();
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

        let mut ai = get_nn::<f32, HIDDEN_NEURONS>();

        let mut rng = rand::thread_rng();
        let mut training_data = gen_data(&mut rng, TRAINING_DATA_COUNT).collect::<Vec<_>>();

        for _ in 1..=EPOCHS {
            ai.train_single_thread(&training_data);
            training_data.shuffle(&mut rng);
        }

        for p in gen_data(&mut rng, 30) {
            let (_, loss) = ai.test(&p).into_tuple();
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
            let mut ai = get_nn::<f32, $neurons>();
            let mut rng = rand::thread_rng();
            let mut training_data = gen_data(&mut rng, $data_count).collect::<Vec<_>>();
            b.iter(|| {
                black_box(NNTrainer::train_rayon(
                    black_box(&mut ai),
                    black_box(&training_data),
                ));
                training_data.shuffle(&mut rng);
            })
        }
    )* };
}
pub use _bench_example_epoch as bench_example_epoch;

/// # Results (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:     105,449 ns/iter (+/- 8,709)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     421,027 ns/iter (+/- 375,106)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     697,514 ns/iter (+/- 39,467)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     520,097 ns/iter (+/- 33,921)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   2,027,946 ns/iter (+/- 267,714)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   3,461,679 ns/iter (+/- 344,771)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     917,680 ns/iter (+/- 10,399)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   3,620,987 ns/iter (+/- 167,507)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   6,214,099 ns/iter (+/- 338,313)
/// ```
///
/// # Results with rayon (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:     154,521 ns/iter (+/- 145,971)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     451,887 ns/iter (+/- 94,258)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     684,421 ns/iter (+/- 148,162)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     395,819 ns/iter (+/- 96,634)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   1,472,725 ns/iter (+/- 405,848)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   1,983,029 ns/iter (+/- 452,357)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     634,512 ns/iter (+/- 118,611)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   1,759,568 ns/iter (+/- 501,777)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   3,660,244 ns/iter (+/- 947,533)
/// ```
///
/// # Results with rayon (f32)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:     156,445 ns/iter (+/- 225,424)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     417,356 ns/iter (+/- 88,478)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     731,473 ns/iter (+/- 170,163)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     411,928 ns/iter (+/- 57,390)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   1,360,821 ns/iter (+/- 308,288)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   2,323,408 ns/iter (+/- 503,524)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     581,157 ns/iter (+/- 113,416)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   2,481,336 ns/iter (+/- 855,139)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   3,320,163 ns/iter (+/- 956,815)
/// ```
///
/// -----
///
/// # Results const_tensor single thread (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:      65,421 ns/iter (+/- 21,530)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     251,345 ns/iter (+/- 11,503)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     419,846 ns/iter (+/- 22,974)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     322,083 ns/iter (+/- 3,204)
/// test benches::epoch_datacount_500_neurons_500 ... bench:   1,236,425 ns/iter (+/- 74,324)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   2,088,645 ns/iter (+/- 112,973)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     618,036 ns/iter (+/- 206,855)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   2,238,599 ns/iter (+/- 95,940)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   3,750,461 ns/iter (+/- 242,631)
/// ```
///
/// # Results const_tensor rayon (f64)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:      90,561 ns/iter (+/- 19,783)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     204,930 ns/iter (+/- 26,895)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     523,425 ns/iter (+/- 160,540)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     210,831 ns/iter (+/- 51,745)
/// test benches::epoch_datacount_500_neurons_500 ... bench:     741,334 ns/iter (+/- 122,110)
/// test benches::epoch_datacount_500_neurons_900 ... bench:   1,121,656 ns/iter (+/- 229,572)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     364,944 ns/iter (+/- 108,910)
/// test benches::epoch_datacount_900_neurons_500 ... bench:   1,032,378 ns/iter (+/- 270,824)
/// test benches::epoch_datacount_900_neurons_900 ... bench:   1,686,858 ns/iter (+/- 233,635)
/// ```
///
/// # Results const_tensor rayon (f32)
///
/// ```
/// test benches::epoch_datacount_100_neurons_100 ... bench:      65,585 ns/iter (+/- 8,662)
/// test benches::epoch_datacount_100_neurons_500 ... bench:     136,510 ns/iter (+/- 17,128)
/// test benches::epoch_datacount_100_neurons_900 ... bench:     203,447 ns/iter (+/- 32,356)
/// test benches::epoch_datacount_500_neurons_100 ... bench:     181,386 ns/iter (+/- 27,219)
/// test benches::epoch_datacount_500_neurons_500 ... bench:     371,093 ns/iter (+/- 83,309)
/// test benches::epoch_datacount_500_neurons_900 ... bench:     702,488 ns/iter (+/- 226,175)
/// test benches::epoch_datacount_900_neurons_100 ... bench:     243,149 ns/iter (+/- 32,538)
/// test benches::epoch_datacount_900_neurons_500 ... bench:     620,403 ns/iter (+/- 783,641)
/// test benches::epoch_datacount_900_neurons_900 ... bench:     914,991 ns/iter (+/- 220,716)
/// ```
#[cfg(test)]
mod benches {
    use super::*;

    extern crate test;
    use test::*;

    bench_example_epoch! {
        epoch_datacount_100_neurons_100: 100, 100;
        epoch_datacount_100_neurons_500: 500, 100;
        epoch_datacount_100_neurons_900: 900, 100;

        epoch_datacount_500_neurons_100: 100, 500;
        epoch_datacount_500_neurons_500: 500, 500;
        epoch_datacount_500_neurons_900: 900, 500;

        epoch_datacount_900_neurons_100: 100, 900;
        epoch_datacount_900_neurons_500: 500, 900;
        epoch_datacount_900_neurons_900: 900, 900;
    }
}
