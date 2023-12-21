//! # Clip gradient norm module

use crate::{Gradient, Norm, ParamsIter};
use serde::{Deserialize, Serialize};

/// Used to clip the norm of a [`Gradient`] to a maximum value of `max_norm`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipGradientNorm {
    /// maximum value allowed for the norm of the gradient
    pub max_norm: f64,
    /// norm type used for the calculations
    pub norm_type: Norm,
}

impl ClipGradientNorm {
    /// Creates a new `ClipGradientNorm`.
    pub fn new(norm_type: Norm, max_norm: f64) -> Self {
        Self { norm_type, max_norm }
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    pub fn clip_gradient(self, gradient: &mut Gradient) {
        let iter = gradient.iter().copied();
        let norm = self.norm_type.calculate(iter);
        if norm > self.max_norm {
            let clip_factor = self.max_norm / (norm + 1e-6); // 1e-6 copied from: <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
            gradient.iter_mut().for_each(|x| *x *= clip_factor);
        }
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    ///
    /// see <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
    pub fn clip_gradient_pytorch(self, gradient: &mut Gradient) {
        let iter = gradient.iter().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = self.max_norm / (norm + 1e-6);
        if clip_factor >= 1.0 {
            // <=> self.max_norm >= norm + 1e-6
            return;
        }
        gradient.iter_mut().for_each(|x| *x *= clip_factor);
    }

    /// Clips the norm of the `gradient` to a maximum value of `self.max_norm`.
    ///
    /// see <https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_>
    pub fn clip_gradient_pytorch_device(self, gradient: &mut Gradient) {
        let iter = gradient.iter().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = (self.max_norm / (norm + 1e-6)).min(1.0);
        gradient.iter_mut().for_each(|x| *x *= clip_factor);
    }
}
