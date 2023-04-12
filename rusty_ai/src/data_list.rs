use crate::data::DataPair;
use itertools::Itertools;
use std::{ops::Index, vec::IntoIter};

#[derive(Debug, Clone)]
pub struct PairList<const IN: usize, const OUT: usize>(pub Vec<DataPair<IN, OUT>>);

// From

impl<const IN: usize, const OUT: usize> From<Vec<DataPair<IN, OUT>>> for PairList<IN, OUT> {
    fn from(value: Vec<DataPair<IN, OUT>>) -> Self {
        Self(value)
    }
}

impl<const IN: usize, const OUT: usize> From<DataPair<IN, OUT>> for PairList<IN, OUT> {
    fn from(value: DataPair<IN, OUT>) -> Self {
        Self(vec![value])
    }
}

impl<const IN: usize, const OUT: usize> From<Vec<([f64; IN], [f64; OUT])>> for PairList<IN, OUT> {
    fn from(value: Vec<([f64; IN], [f64; OUT])>) -> Self {
        value.into_iter().map(DataPair::from).collect_vec().into()
    }
}

impl From<Vec<(f64, f64)>> for PairList<1, 1> {
    fn from(value: Vec<(f64, f64)>) -> Self {
        value.into_iter().map(DataPair::from).collect_vec().into()
    }
}

impl PairList<1, 1> {
    pub fn from_simple_vecs(vec_in: Vec<f64>, vec_out: Vec<f64>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }
}

impl<const IN: usize, const OUT: usize> PairList<IN, OUT> {
    pub fn from_vecs(vec_in: Vec<[f64; IN]>, vec_out: Vec<[f64; OUT]>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }
}

// Into

impl<const IN: usize, const OUT: usize> IntoIterator for PairList<IN, OUT> {
    type Item = DataPair<IN, OUT>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const IN: usize, const OUT: usize> PairList<IN, OUT> {
    pub fn iter(&self) -> impl IntoIterator<Item = &DataPair<IN, OUT>> {
        self.0.iter()
    }
}

impl<const IN: usize, const OUT: usize> PairList<IN, OUT> {
    pub fn into_iter_tuple(self) -> impl Iterator<Item = ([f64; IN], [f64; OUT])> {
        self.into_iter().map(Into::into)
    }
}

// other

impl<const IN: usize, const OUT: usize> Index<usize> for PairList<IN, OUT> {
    type Output = DataPair<IN, OUT>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
