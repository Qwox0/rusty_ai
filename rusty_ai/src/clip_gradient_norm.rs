use crate::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipGradientNorm {
    pub max_norm: f64,
    pub norm_type: Norm,
}

impl ClipGradientNorm {
    constructor! { pub new -> norm_type: Norm, max_norm: f64 }

    pub fn clip_gradient(&self, gradient: &mut Gradient) {
        let iter = gradient.iter_params().copied();
        let norm = self.norm_type.calculate(iter);
        if norm > self.max_norm {
            let clip_factor = self.max_norm / (norm + 1e-6); // 1e-6 copied from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
            println!("clip_factor: {}", clip_factor);
            gradient.iter_mut_params().for_each(|x| *x *= clip_factor);
        }
    }

    /// [https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_]
    pub fn clip_gradient_pytorch(&self, gradient: &mut Gradient) {
        let iter = gradient.iter_params().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = self.max_norm / (norm + 1e-6);
        if clip_factor >= 1.0 {
            // == self.max_norm >= norm + 1e-6
            return
        }
        gradient.iter_mut_params().for_each(|x| *x *= clip_factor);
    }

    /// [https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_]
    pub fn clip_gradient_pytorch_device(&self, gradient: &mut Gradient) {
        let iter = gradient.iter_params().copied();
        let norm = self.norm_type.calculate(iter);
        let clip_factor = (self.max_norm / (norm + 1e-6)).min(1.0);
        gradient.iter_mut_params().for_each(|x| *x *= clip_factor);
    }
}
