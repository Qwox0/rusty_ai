//! # Clip gradient norm module

use crate::{nn::GradComponent, norm::Norm};
use const_tensor::{Float, Shape, Tensor};
use serde::{Deserialize, Serialize};

/// Used to clip the norm of a [`Gradient`] to a maximum value of `max_norm`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipGradientNorm<X> {
    /// maximum value allowed for the norm of the gradient
    pub max_norm: X,
    /// norm type used for the calculations
    pub norm_type: Norm,
}

impl<F: Float> ClipGradientNorm<F> {
    /// Creates a new `ClipGradientNorm`.
    pub fn new(norm_type: Norm, max_norm: F) -> Self {
        Self { norm_type, max_norm }
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    pub fn clip_gradient<Grad: GradComponent<F>>(self, gradient: &mut Grad) {
        let iter = gradient.iter_param().copied();
        let norm = self.norm_type.calculate(iter);
        if norm > self.max_norm {
            let clip_factor = self.max_norm / (norm + F::f_lit(1e-6)); // 1e-6 copied from: <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
            gradient.iter_param_mut().for_each(|x| *x *= clip_factor.cast());
        }
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    ///
    /// see <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
    pub fn clip_gradient_pytorch<Grad: GradComponent<F>>(self, gradient: &mut Grad) {
        let iter = gradient.iter_param().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = self.max_norm / (norm + F::f_lit(1e-6));
        if clip_factor >= F::ONE {
            // <=> self.max_norm >= norm + 1e-6
            return;
        }
        gradient.iter_param_mut().for_each(|x| *x *= clip_factor.cast());
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    ///
    /// see <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
    pub fn clip_gradient_pytorch_device<Grad: GradComponent<F>>(self, gradient: &mut Grad) {
        let iter = gradient.iter_param().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = (self.max_norm / (norm + F::f_lit(1e-6))).min(F::ONE);
        gradient.iter_param_mut().for_each(|x| *x *= clip_factor.cast());
    }
}
