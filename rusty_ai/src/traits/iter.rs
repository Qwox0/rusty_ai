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
        self.iter_layers().map(IterParams::iter_parameters).flatten()
    }

    fn iter_mut_parameters<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f64> {
        self.iter_mut_layers().map(IterParams::iter_mut_parameters).flatten()
    }
}

mod macros {
    /// impl_IterParams! { $ty:ty : $weights:ident , $bias:ident }
    macro_rules! impl_IterParams {
        ($ty:ty : $weights:ident, $bias:ident) => {
            impl $crate::prelude::IterParams for $ty {
                fn iter_weights<'a>(&'a self) -> impl Iterator<Item = &'a f64> {
                    self.$weights.iter()
                }

                fn iter_bias<'a>(&'a self) -> impl Iterator<Item = &'a f64> { self.$bias.iter() }

                fn iter_mut_parameters<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f64> {
                    self.$weights.iter_mut().chain(self.$bias.iter_mut())
                }
            }
        };
    }

    pub(crate) use impl_IterParams;
}

pub(crate) use macros::impl_IterParams;
