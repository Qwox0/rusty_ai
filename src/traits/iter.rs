/*
use matrix::Element;

/// trait for iterating over the parameters of type [`f64`].
pub trait ParamsIter<X: Element> {
    /// creates an [`Iterator`] over references to the parameters of `self`
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a X>;
    /// creates an [`Iterator`] over mutable references to the parameters of `self`
    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut X>;
}

pub(crate) fn default_params_chain<W, B, T>(
    weights: W,
    bias: B,
) -> impl DoubleEndedIterator<Item = T>
where
    W: IntoIterator<Item = T>,
    W::IntoIter: DoubleEndedIterator,
    B: IntoIterator<Item = T>,
    B::IntoIter: DoubleEndedIterator,
{
    weights.into_iter().chain(bias)
}

/*
pub trait LayerIter<'a> {
    type Layer: 'a + ParamsIter<'a>;

    type Iter: 'a + Iterator<Item = &'a Self::Layer>;
    type IterMut: 'a + Iterator<Item = &'a mut Self::Layer>;

    fn iter_layers(&'a self) -> Self::Iter;
    fn iter_mut_layers(&'a mut self) -> Self::IterMut;

    fn iter_params(&'a self) -> impl Iterator<Item = &'a f64>
    where Self::Layer: ParamsIter<'a> {
        self.iter_layers().map(ParamsIter::iter_params).flatten()
    }

    fn iter_mut_params(&'a mut self) -> impl Iterator<Item = &'a mut f64>
    where Self::Layer: ParamsIter<'a> {
        self.iter_mut_layers().map(ParamsIter::iter_mut_params).flatten()
    }
}

mod macros {
    /// impl_IterParams! { $ty:ty : $weights:ident , $bias:ident }
    macro_rules! impl_IterParams {
        ($ty:ty : $weights:ident, $bias:ident) => {
            impl $crate::ParamsIter for $ty {
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
*/
