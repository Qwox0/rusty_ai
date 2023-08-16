use crate::prelude::*;
use std::iter::once;

/// Contains the estimated Gradient of the Costfunction with respect to the
/// weights and the bias of a layer in
#[derive(Debug, Clone)]
pub struct GradientLayer {
    pub weight_gradient: WeightGradient,
    pub bias_gradient: BiasGradient,
}

impl GradientLayer {
    constructor! { pub new -> weight_gradient: Matrix<f64>, bias_gradient: LayerBias }

    pub fn iter_mut_neurons<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (&'a mut Vec<f64>, &'a mut f64)> {
        self.weight_gradient.iter_rows_mut().zip(self.bias_gradient.iter_mut())
    }
}

impl<'a> ParamsIter<'a> for GradientLayer {
    fn iter_parameters(&'a self) -> impl Iterator<Item = &'a f64> {
        Self::default_chain(self.weight_gradient.iter(), self.bias_gradient.iter())
    }

    fn iter_mut_parameters(&'a mut self) -> impl Iterator<Item = &'a mut f64> {
        Self::default_chain(self.weight_gradient.iter_mut(), self.bias_gradient.iter_mut())
    }
}

impl std::fmt::Display for GradientLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bias_header = "Biases:".to_string();
        let bias_str_iter =
            once(bias_header).chain(self.bias_gradient.iter().map(ToString::to_string));
        let bias_column_width = bias_str_iter.clone().map(|s| s.len()).max().unwrap_or(0);
        let mut bias_lines = bias_str_iter.map(|s| format!("{s:^bias_column_width$}"));
        for (idx, l) in self.weight_gradient.to_string_with_title("Weights:")?.lines().enumerate() {
            if idx != 0 {
                write!(f, "\n")?;
            }
            write!(f, "{} {}", l, bias_lines.next().unwrap_or_default())?;
        }
        Ok(())
    }
}
