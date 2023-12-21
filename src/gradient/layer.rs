use crate::{
    bias::LayerBias,
    gradient::aliases::{BiasGradient, WeightGradient},
    matrix::Matrix,
    traits::default_params_chain,
    ParamsIter,
};
#[allow(unused_imports)]
use crate::{layer::Layer, Gradient};
use serde::{Deserialize, Serialize};
use std::iter::once;

/// Contains the estimated Gradient of the loss function with respect to the
/// weights and the bias in a layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientLayer {
    pub(super) weight_gradient: WeightGradient,
    pub(super) bias_gradient: BiasGradient,
}

impl GradientLayer {
    /// Creates part of a [`Gradient`] which represents the derivatives with respect to the
    /// parameters in a [`Layer`].
    pub fn new(weight_gradient: Matrix<f64>, bias_gradient: LayerBias) -> Self {
        Self { weight_gradient, bias_gradient }
    }

    /// Iterates through the parameters of the layer mutably.
    pub fn iter_mut_neurons<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (&'a mut Vec<f64>, &'a mut f64)> {
        self.weight_gradient.iter_rows_mut().zip(&mut self.bias_gradient)
    }
}

impl ParamsIter for GradientLayer {
    fn iter<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a f64> {
        default_params_chain(&self.weight_gradient, &self.bias_gradient)
    }

    fn iter_mut<'a>(&'a mut self) -> impl DoubleEndedIterator<Item = &'a mut f64> {
        default_params_chain(&mut self.weight_gradient, &mut self.bias_gradient)
    }
}

impl std::fmt::Display for GradientLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bias_header = "Biases:".to_string();
        let bias_str_iter =
            once(bias_header).chain(self.bias_gradient.iter().map(ToString::to_string));
        // Todo
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

#[cfg(test)]
mod tests {
    use crate::{
        neural_network::builder::{BuildLayer, NNBuilder},
        Initializer,
    };

    #[test]
    fn display_gradient_layer() {
        let nn = NNBuilder::default()
            .input::<3>()
            .layer(3, Initializer::PytorchDefault, Initializer::Ones())
            .build::<3>();
        let layer = &nn.get_layers()[0];
        println!("{}", layer);
        let grad = layer.init_zero_gradient();
        println!("{}", grad);
    }
}
