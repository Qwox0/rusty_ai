use std::iter::once;

use super::aliases::{BiasGradient, WeightGradient};
use crate::{
    layer::{AddBias, LayerBias},
    matrix::Matrix,
    util::{
        constructor, EntryAdd, EntryDiv, EntryMul, EntrySub, Lerp, ScalarAdd, ScalarDiv, ScalarMul,
        ScalarSub,
    },
};

/// Contains the estimated Gradient of the Costfunction with respect to the weights and the bias of
/// a layer in
#[derive(Debug, Clone)]
pub struct GradientLayer {
    pub bias_gradient: BiasGradient,
    pub weight_gradient: WeightGradient,
}

impl GradientLayer {
    constructor! { pub new -> weight_gradient: Matrix<f64>, bias_gradient: LayerBias }

    pub fn add_next_backpropagation(
        &mut self,
        next_weights_change: WeightGradient,
        next_bias_change: BiasGradient,
    ) {
        self.bias_gradient.add_bias_mut(&next_bias_change);
        self.weight_gradient.add_entries_mut(&next_weights_change);
    }

    pub fn sqare_entries(mut self) -> GradientLayer {
        self.bias_gradient.sqare_entries_mut();
        self.weight_gradient.iter_mut().for_each(|x| *x *= *x);
        self
    }

    pub fn sqrt_entries(mut self) -> GradientLayer {
        self.bias_gradient.sqrt_entries_mut();
        self.weight_gradient.iter_mut().for_each(|x| *x = x.sqrt());
        self
    }

    pub fn iter_numbers(&self) -> impl Iterator<Item = &f64> {
        let iter = self.weight_gradient.iter();
        iter.chain(self.bias_gradient.iter_numbers())
    }
}

macro_rules! impl_entrywise_arithmetic {
    ( $trait:ident : $fn:ident ) => {
        impl $trait<&GradientLayer> for GradientLayer {
            fn $fn(&mut self, rhs: &GradientLayer) -> &mut Self {
                self.bias_gradient.$fn(&rhs.bias_gradient);
                self.weight_gradient.$fn(&rhs.weight_gradient);
                self
            }
        }
    };
}

impl_entrywise_arithmetic! { EntryAdd: add_entries_mut }
impl_entrywise_arithmetic! { EntrySub: sub_entries_mut }
impl_entrywise_arithmetic! { EntryMul: mul_entries_mut }
impl_entrywise_arithmetic! { EntryDiv: div_entries_mut }

macro_rules! impl_scalar_arithmetic {
    ( $trait:ident : $fn:ident ) => {
        impl $trait for GradientLayer {
            fn $fn(&mut self, scalar: f64) -> &mut Self {
                self.bias_gradient.$fn(scalar);
                self.weight_gradient.$fn(scalar);
                self
            }
        }
    };
}

impl_scalar_arithmetic! { ScalarAdd : add_scalar_mut }
impl_scalar_arithmetic! { ScalarMul : mul_scalar_mut }
impl_scalar_arithmetic! { ScalarDiv : div_scalar_mut }

impl Lerp<&GradientLayer> for GradientLayer {
    fn lerp_mut(&mut self, other: &GradientLayer, blend: f64) -> &mut Self {
        self.bias_gradient.lerp_mut(&other.bias_gradient, blend);
        self.weight_gradient.lerp_mut(&other.weight_gradient, blend);
        self
    }
}
