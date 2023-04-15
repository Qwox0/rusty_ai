use crate::data::DataPair;
use itertools::Itertools;
use rand::{
    seq::{SliceChooseIter, SliceRandom},
    Rng,
};
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct DataList<const IN: usize, const OUT: usize>(pub Vec<DataPair<IN, OUT>>);

// From

impl<const IN: usize, const OUT: usize> From<Vec<DataPair<IN, OUT>>> for DataList<IN, OUT> {
    fn from(value: Vec<DataPair<IN, OUT>>) -> Self {
        Self(value)
    }
}

impl<const IN: usize, const OUT: usize> From<DataPair<IN, OUT>> for DataList<IN, OUT> {
    fn from(value: DataPair<IN, OUT>) -> Self {
        Self(vec![value])
    }
}

impl<const IN: usize, const OUT: usize> From<Vec<([f64; IN], [f64; OUT])>> for DataList<IN, OUT> {
    fn from(value: Vec<([f64; IN], [f64; OUT])>) -> Self {
        value.into_iter().map(DataPair::from).collect_vec().into()
    }
}

impl<const IN: usize, const OUT: usize> DataList<IN, OUT> {
    pub fn from_vecs(vec_in: Vec<[f64; IN]>, vec_out: Vec<[f64; OUT]>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }

    pub fn random(amount: usize, get_out: impl Fn([f64; IN]) -> [f64; OUT]) -> DataList<IN, OUT> {
        let pairs: Vec<_> = (0..amount)
            .into_iter()
            .map(|_| DataPair::random(&get_out))
            .collect();
        DataList::from(pairs)
    }

    pub fn iter(&self) -> impl Iterator<Item = &DataPair<IN, OUT>> {
        self.0.iter()
    }

    pub fn into_iter_tuple(self) -> impl Iterator<Item = ([f64; IN], [f64; OUT])> {
        self.into_iter().map(Into::into)
    }

    pub fn choose_multiple<R>(
        &self,
        rng: &mut R,
        amount: usize,
    ) -> impl Iterator<Item = &DataPair<IN, OUT>>
    where
        R: Rng + ?Sized,
    {
        self.0.choose_multiple(rng, amount)
    }
}

impl<const IN: usize, const OUT: usize> IntoIterator for DataList<IN, OUT> {
    type Item = DataPair<IN, OUT>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const IN: usize, const OUT: usize> Index<usize> for DataList<IN, OUT> {
    type Output = DataPair<IN, OUT>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// Simple (IN == OUT == 1)

impl From<Vec<(f64, f64)>> for DataList<1, 1> {
    fn from(value: Vec<(f64, f64)>) -> Self {
        value.into_iter().map(DataPair::from).collect_vec().into()
    }
}

impl DataList<1, 1> {
    pub fn from_simple_vecs(vec_in: Vec<f64>, vec_out: Vec<f64>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }

    pub fn random_simple(amount: usize, get_out: impl Fn(f64) -> f64) -> DataList<1, 1> {
        let pairs: Vec<_> = (0..amount)
            .into_iter()
            .map(|_| DataPair::random_simple(&get_out))
            .collect();
        DataList::from(pairs)
    }

    pub fn into_iter_tuple_simple(self) -> impl Iterator<Item = (f64, f64)> {
        self.into_iter().map(Into::into)
    }

    pub fn iter_tuple_simple(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.iter().map(DataPair::simple_tuple)
    }
}
