use crate::{data::Pair, export::ExportToJs};
use itertools::Itertools;
use rand::{distributions::uniform::SampleRange, seq::SliceRandom, Rng};
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct PairList<const IN: usize, const OUT: usize>(pub Vec<Pair<IN, OUT>>);

// From

impl<const IN: usize, const OUT: usize> From<Vec<Pair<IN, OUT>>> for PairList<IN, OUT> {
    fn from(value: Vec<Pair<IN, OUT>>) -> Self {
        Self(value)
    }
}

impl<const IN: usize, const OUT: usize> From<Pair<IN, OUT>> for PairList<IN, OUT> {
    fn from(value: Pair<IN, OUT>) -> Self {
        Self(vec![value])
    }
}

impl<const IN: usize, const OUT: usize> From<Vec<([f64; IN], [f64; OUT])>> for PairList<IN, OUT> {
    fn from(value: Vec<([f64; IN], [f64; OUT])>) -> Self {
        value.into_iter().map(Pair::from).collect_vec().into()
    }
}

impl<const IN: usize, const OUT: usize> PairList<IN, OUT> {
    pub fn from_vecs(vec_in: Vec<[f64; IN]>, vec_out: Vec<[f64; OUT]>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Pair<IN, OUT>> {
        self.0.iter()
    }

    pub fn into_iter_tuple(self) -> impl Iterator<Item = ([f64; IN], [f64; OUT])> {
        self.into_iter().map(Into::into)
    }

    pub fn choose_multiple<R>(
        &self,
        rng: &mut R,
        amount: usize,
    ) -> impl Iterator<Item = &Pair<IN, OUT>>
    where
        R: Rng + ?Sized,
    {
        self.0.choose_multiple(rng, amount)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[Pair<IN, OUT>] {
        self.0.as_slice()
    }
}

impl<const IN: usize, const OUT: usize> IntoIterator for PairList<IN, OUT> {
    type Item = Pair<IN, OUT>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const IN: usize, const OUT: usize> Index<usize> for PairList<IN, OUT> {
    type Output = Pair<IN, OUT>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// Simple (IN == OUT == 1)

impl From<Vec<(f64, f64)>> for PairList<1, 1> {
    fn from(value: Vec<(f64, f64)>) -> Self {
        value.into_iter().map(Pair::from).collect_vec().into()
    }
}

impl PairList<1, 1> {
    pub fn from_simple_vecs(vec_in: Vec<f64>, vec_out: Vec<f64>) -> Self {
        vec_in.into_iter().zip(vec_out).collect_vec().into()
    }

    pub fn random_simple(
        amount: usize,
        range: impl SampleRange<f64> + Clone,
        get_out: impl Fn(f64) -> f64,
    ) -> PairList<1, 1> {
        let pairs: Vec<_> = (0..amount)
            .into_iter()
            .map(|_| Pair::random_simple(range.clone(), &get_out))
            .collect();
        PairList::from(pairs)
    }

    pub fn into_iter_tuple_simple(self) -> impl Iterator<Item = (f64, f64)> {
        self.into_iter().map(Into::into)
    }

    pub fn iter_tuple_simple(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.iter().map(Pair::simple_tuple)
    }
}

impl ExportToJs for PairList<1, 1> {
    fn get_js_value(&self) -> String {
        let (x, y): (Vec<f64>, Vec<f64>) = self.clone().into_iter_tuple_simple().unzip();
        format!("{{ x: {}, y: {} }}", x.get_js_value(), y.get_js_value())
    }
}
