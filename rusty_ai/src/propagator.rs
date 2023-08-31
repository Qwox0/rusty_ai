use crate::{
    optimizer::sgd::{SGD, SGD_},
    prelude::{NLLLoss, NNTrainer, Pair},
};

pub trait SimplePropagator<const IN: usize, const OUT: usize> {
    fn propagate_arr(&self, input: &[f64; IN]) -> [f64; OUT];
}

pub trait NNInput<const IN: usize> {
    fn get_input(&self) -> &[f64; IN];
}

impl<const IN: usize> NNInput<IN> for [f64; IN] {
    fn get_input(&self) -> &[f64; IN] {
        self
    }
}

impl<const IN: usize> NNInput<IN> for &[f64; IN] {
    fn get_input(&self) -> &[f64; IN] {
        self
    }
}

fn test() {
    let ai: NNTrainer<3, 5, NLLLoss, _> = NNTrainer::default::<SGD>();

    let i = (&[1.618033987; 3], &1);
    let batch = std::iter::once(i);

    //let a = ai.backpropagate(batch);

    //asdf(batch);
}

fn asdf<'a, const IN: usize, B>(input: B)
where
    B: IntoIterator + 'a,
    B::Item: Into<&'a [f64; IN]>,
{
}
