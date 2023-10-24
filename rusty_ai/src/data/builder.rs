use super::value_list::ValueList;
use crate::util::RngWrapper;
use rand::{distributions::Uniform, prelude::Distribution, Rng};
use std::ops::Range;

pub struct DataBuilder<D: Distribution<f64>> {
    distr: D,
    rng_seed: Option<u64>,
}

//pub struct DataIteratorBuilder { }

impl Default for DataBuilder<Uniform<f64>> {
    fn default() -> Self {
        DataBuilder { distr: Uniform::from(0.0..1.0), rng_seed: None }
    }
}

impl DataBuilder<Uniform<f64>> {
    pub fn uniform(range: Range<f64>) -> DataBuilder<Uniform<f64>> {
        DataBuilder::with_distr(Uniform::from(range))
    }
}

impl<D: Distribution<f64>> DataBuilder<D> {
    pub fn with_distr(distr: D) -> DataBuilder<D> {
        DataBuilder { distr, ..Default::default() }
    }

    pub fn distr<ND: Distribution<f64>>(self, distr: ND) -> DataBuilder<ND> {
        DataBuilder { distr, ..self }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        let _ = self.rng_seed.insert(seed);
        self
    }

    fn make_rng(&self) -> RngWrapper {
        RngWrapper::new(self.rng_seed)
    }

    pub fn build<const IN: usize>(&self, data_count: usize) -> ValueList<IN> {
        self.make_rng()
            .sample_iter(&self.distr)
            .array_chunks()
            .take(data_count)
            .collect::<Vec<_>>()
            .into()
    }

    //pub fn build_single(&self) -> Pair {}
}
