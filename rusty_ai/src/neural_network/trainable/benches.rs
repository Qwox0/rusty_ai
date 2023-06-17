extern crate test;

use crate::{data::DataBuilder, prelude::*};
use rayon::prelude::*;
use std::{sync::Mutex, thread::ScopedJoinHandle};
use test::black_box;

fn calc_grad(nn: &mut TrainableNeuralNetwork<1, 1>, data: &[Pair<1, 1>]) {
    for pair in data {
        let (input, expected_output) = pair.into();
        nn.backpropagation(nn.verbose_propagate(input), expected_output);
    }
}

trait BackpropBenches {
    fn single_thread(&mut self, data: &[Pair<1, 1>]);
    fn single_thread2(&mut self, data: &[Pair<1, 1>]);
    fn arc_mutex(&mut self, data: &[Pair<1, 1>]);
    fn mpsc_out(&mut self, data: &[Pair<1, 1>]);
    fn spmc_in(&mut self, data: &[Pair<1, 1>]);
    fn rayon_iter(&mut self, data: &[Pair<1, 1>]);
}

/*
impl BackpropBenches for TrainableNeuralNetwork<1, 1> {
    fn single_thread(&mut self, data: &[Pair<1, 1>]) {
        let data_count = data.len();
        let mut gradient = self.network.init_zero_gradient();
        for (input, expected_output) in data.iter().map(Into::into) {
            self.network.backpropagation(
                self.verbose_propagate(input),
                expected_output,
                &mut gradient,
            );
        }
        gradient.div_scalar_mut(data_count as f64);
        self.optimize();
    }

    fn single_thread2(&mut self, data: &[Pair<1, 1>]) {
        let data_count = data.len();
        let mut gradient = calc_grad(self, data);
        gradient.div_scalar_mut(data_count as f64);
        self.optimize();
    }

    fn arc_mutex(&mut self, data: &[Pair<1, 1>]) {
        let data_count = data.len();
        let cpus = crate::util::cpu_count();
        let chunk_size = (data_count as f64 / cpus as f64).ceil() as usize;
        let gradient = Mutex::new(self.network.init_zero_gradient());
        std::thread::scope(|scope| {
            for chunk in data.chunks(chunk_size) {
                scope.spawn(|| {
                    let grad = calc_grad(self, chunk);
                    gradient.lock().unwrap().add_entries_mut(grad);
                });
            }
        });
        todo!("update weigths")
    }

    /*
    fn join(rx: Receiver<Option<usize>>, max: usize) -> Option<usize> {
        let mut found = 0;
        while let Ok(x) = rx.recv() {
            if x.is_some() {
                return x;
            }
            found += 1;
            if found == max {
                return None;
            }
        }

        return None;
    }
    fn my_multithread(data: &'static [u8]) -> Option<usize> {
        let regions = (data.len() / CPUS) + 1; // +1: round up instead of down
        let (tx, rx) = std::sync::mpsc::channel();
        for (idx, chunk) in data.chunks(regions).enumerate() {
            let inner_tx = tx.clone();
            std::thread::spawn(move || {
                _ = inner_tx.send(david_a_perez(chunk).map(|x| x + regions * idx));
            });
        }

        return join(rx, CPUS);
    }
    */
    fn mpsc_out(&mut self, data: &[Pair<1, 1>]) {
        //let (send, recv) = std::sync::mpsc::channel();

        let data_count = data.len();
        let cpus = crate::util::cpu_count();
        let chunk_size = (data_count as f64 / cpus as f64).ceil() as usize;

        std::thread::scope(|scope| {
            for chunk in data.chunks(chunk_size) {
                scope.spawn(|| {
                    let grad = calc_grad(self, chunk);
                });
            }
        });
        todo!()
    }

    fn spmc_in(&mut self, data: &[Pair<1, 1>]) {
        let data_count = data.len();
        let cpus = crate::util::cpu_count();
        let (mut send, recv) = spmc::channel::<Option<&Pair<1, 1>>>();

        let mut gradient = std::thread::scope(|scope| {
            let nn = &self;
            let handles = (0..cpus)
                .map(|_| {
                    let recv = recv.clone();
                    scope.spawn(move || {
                        let mut gradient = nn.network.init_zero_gradient();
                        while let Some(pair) = recv.recv().unwrap() {
                            let (input, expected_output) = pair.into();
                            nn.network.backpropagation(
                                nn.verbose_propagate(input),
                                expected_output,
                                &mut gradient,
                            );
                        }
                        gradient
                    })
                })
                .collect::<Vec<_>>();

            let end_signals = (0..cpus).map(|_| None);
            data.into_iter()
                .map(|p| Some(p))
                .chain(end_signals)
                .for_each(|x| send.send(x).unwrap());

            handles
                .into_iter()
                .map(ScopedJoinHandle::join)
                .map(Result::unwrap)
                .fold(self.network.init_zero_gradient(), |acc, x| {
                    acc.add_entries(x)
                })
        });

        gradient.div_scalar_mut(data_count as f64);
        self.optimize();
    }

    fn rayon_iter(&mut self, data: &[Pair<1, 1>]) {
        let data_count = data.len();
        let mut gradient = data
            .par_iter()
            .map(Into::into)
            .map(|(input, expected_output)| {
                let mut gradient = self.network.init_zero_gradient();
                self.network.backpropagation(
                    self.verbose_propagate(input),
                    expected_output,
                    &mut gradient,
                );
                gradient
            })
            .reduce(
                || self.network.init_zero_gradient(),
                |acc, a| acc.add_entries(a),
            );
        gradient.div_scalar_mut(data_count as f64);
        self.optimize();
    }
}
*/

impl<'a> IntoParallelIterator for &'a PairList<1, 1> {
    type Item = &'a Pair<1, 1>;
    type Iter = rayon::slice::Iter<'a, Pair<1, 1>>;
    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl ParallelSlice<Pair<1, 1>> for PairList<1, 1> {
    fn as_parallel_slice(&self) -> &[Pair<1, 1>] {
        self.0.as_parallel_slice()
    }
}

fn setup(data_count: usize) -> (TrainableNeuralNetwork<1, 1>, PairList<1, 1>) {
    const SEED: u64 = 69420;
    const NEURON_COUNT: usize = 5;
    let ai = NeuralNetworkBuilder::default()
        .rng_seed(SEED)
        .default_activation_function(ActivationFn::default_relu())
        .input()
        .random_layer(NEURON_COUNT)
        .random_layer(NEURON_COUNT)
        .random_layer(NEURON_COUNT)
        .default_activation_function(ActivationFn::Identity)
        .random_layer(1)
        .output()
        .sgd_optimizer(GradientDescent::default())
        .build();

    let data = DataBuilder::uniform(-5.0..5.0)
        .seed(SEED)
        .build(data_count)
        .gen_pairs(|x| [x[0].sin()]);

    (ai, data)
}

macro_rules! make_bench {
    ( $fn:ident ) => {
        mod $fn {
            use super::*;
            #[bench]
            fn bench(b: &mut test::Bencher) {
                let (mut ai, data) = setup(100);
                let data = data.as_slice();

                b.iter(|| black_box(BackpropBenches::$fn(black_box(&mut ai), black_box(&data))))
            }
            #[test]
            fn test() {
                let (mut ai, data) = setup(100);
                let data = data.as_slice();

                BackpropBenches::$fn(&mut ai, &data);

                let res = ai.test_propagate(data);
                assert_eq!(res.error, 7538.635825747491);
            }
        }
    };
}

//make_bench! { single_thread }
/*
make_bench! { single_thread2 }
//make_bench! { arc_mutex }
//make_bench! { mpsc_out }
make_bench! { spmc_in }
make_bench! { rayon_iter }
*/
