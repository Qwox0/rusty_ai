pub trait ParamsIter<'a> {
    fn iter_parameters(&'a self) -> impl Iterator<Item = &'a f64>;
    fn iter_mut_parameters(&'a mut self) -> impl Iterator<Item = &'a mut f64>;

    fn default_chain<T>(
        weights: impl IntoIterator<Item = T>,
        bias: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = T> {
        weights.into_iter().chain(bias)
    }
}

pub trait LayerIter<'a> {
    type Layer: 'a + ParamsIter<'a>;

    type Iter: 'a + Iterator<Item = &'a Self::Layer>;
    type IterMut: 'a + Iterator<Item = &'a mut Self::Layer>;

    fn iter_layers(&'a self) -> Self::Iter;
    fn iter_mut_layers(&'a mut self) -> Self::IterMut;

    fn iter_params(&'a self) -> impl Iterator<Item = &'a f64>
    where Self::Layer: ParamsIter<'a> {
        self.iter_layers().map(ParamsIter::iter_parameters).flatten()
    }

    fn iter_mut_params(&'a mut self) -> impl Iterator<Item = &'a mut f64>
    where Self::Layer: ParamsIter<'a> {
        self.iter_mut_layers().map(ParamsIter::iter_mut_parameters).flatten()
    }
}

/*
mod macros {
    /// impl_IterParams! { $ty:ty : $weights:ident , $bias:ident }
    macro_rules! impl_IterParams {
        ($ty:ty : $weights:ident, $bias:ident) => {
            impl $crate::prelude::ParamsIter for $ty {
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
*/
