use crate::{gradient::Gradient, prelude::*};

pub trait Propagator<const IN: usize, const OUT: usize> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT>;

    fn propagate_many(&self, input_list: &Vec<[f64; IN]>) -> Vec<PropagationResult<OUT>>;

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT>;
}

pub trait Trainable<const IN: usize, const OUT: usize> {
    type Trainee;

    fn train(
        &mut self,
        training_data: &PairList<IN, OUT>,
        training_amount: usize,
        epoch_count: usize,
        callback: impl FnMut(usize, &Self::Trainee),
    );

    /// Trains the neural network for one generation/epoch. Uses a small data set `data_pairs` to
    /// find an aproximation for the weights gradient. The neural network's Optimizer changes the
    /// weights by using the calculated gradient.
    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>);
}

pub trait IterParams {
    fn iter_weights<'a>(&'a self) -> impl Iterator<Item = &'a f64>;
    fn iter_bias<'a>(&'a self) -> impl Iterator<Item = &'a f64>;

    fn iter_parameters<'a>(&'a self) -> impl Iterator<Item = &'a f64> {
        self.iter_weights().chain(self.iter_bias())
    }

    fn iter_mut_parameters<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f64>;
}

pub trait IterLayerParams {
    type Layer: IterParams;
    fn iter_layers<'a>(&'a self) -> impl Iterator<Item = &'a Self::Layer>;
    fn iter_mut_layers<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Layer>;

    fn iter_parameters<'a>(&'a self) -> impl Iterator<Item = &'a f64> {
        self.iter_layers()
            .map(IterParams::iter_parameters)
            .flatten()
    }

    fn iter_mut_parameters<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f64> {
        self.iter_mut_layers()
            .map(IterParams::iter_mut_parameters)
            .flatten()
    }
}

mod macros {
    /// impl_IterParams! { $ty:ty : $weights:ident , $bias:ident }
    macro_rules! impl_IterParams {
        ( $ty:ty : $weights:ident , $bias:ident ) => {
            impl IterParams for $ty {
                fn iter_weights<'a>(&'a self) -> impl Iterator<Item = &'a f64> {
                    self.$weights.iter()
                }

                fn iter_bias<'a>(&'a self) -> impl Iterator<Item = &'a f64> {
                    self.$bias.iter()
                }

                fn iter_mut_parameters<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f64> {
                    self.$weights.iter_mut().chain(self.$bias.iter_mut())
                }
            }
        };
    }

    pub(crate) use impl_IterParams;
}

pub(crate) use macros::impl_IterParams;
