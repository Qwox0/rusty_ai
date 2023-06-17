use std::iter::once;

use super::aliases::{BiasGradient, WeightGradient, WeightedSumGradient};
use crate::{
    layer::LayerBias,
    matrix::Matrix,
    traits::{impl_IterParams, IterParams},
    util::{constructor, ScalarMul},
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

    pub fn from_backpropagation(
        weighted_sum_gradient: WeightedSumGradient,
        input: Vec<f64>,
    ) -> GradientLayer {
        let layer_input_count = input.len();
        let layer_neuron_count = weighted_sum_gradient.len();
        let bias_gradient: BiasGradient = LayerBias::from(weighted_sum_gradient.clone());

        let mut weight_gradient: WeightGradient =
            Matrix::new_empty(layer_input_count, layer_neuron_count);
        for &neuron in weighted_sum_gradient.iter() {
            weight_gradient.push_row(input.clone().mul_scalar(neuron));
        }
        GradientLayer::new(weight_gradient, bias_gradient)
    }

    pub fn iter_mut_neurons<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (&'a mut Vec<f64>, &'a mut f64)> {
        self.weight_gradient
            .iter_rows_mut()
            .zip(self.bias_gradient.iter_mut())
    }
}

impl_IterParams! { GradientLayer: weight_gradient, bias_gradient }

impl std::fmt::Display for GradientLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bias_header = "Biases:".to_string();
        let bias_str_iter =
            once(bias_header).chain(self.bias_gradient.iter().map(ToString::to_string));
        let bias_column_width = bias_str_iter.clone().map(|s| s.len()).max().unwrap_or(0);
        let mut bias_lines = bias_str_iter.map(|s| format!("{s:^bias_column_width$}"));
        for (idx, l) in self
            .weight_gradient
            .to_string_with_title("Weights:")?
            .lines()
            .enumerate()
        {
            if idx != 0 {
                write!(f, "\n")?;
            }
            write!(f, "{} {}", l, bias_lines.next().unwrap_or_default())?;
        }
        Ok(())
    }
}
