mod adam;
mod gradient_descent;

use crate::{gradient::Gradient, layer::Layer, nn::NeuralNetwork};
pub use adam::Adam;
use enum_dispatch::enum_dispatch;
pub use gradient_descent::GradientDescent;

pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

#[enum_dispatch]
pub(crate) trait IsOptimizer {
    fn optimize_weights<const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    );
}

#[derive(Debug)]
#[enum_dispatch(IsOptimizer)]
pub enum Optimizer {
    GradientDescent(GradientDescent),
    Adam(Adam),
}

impl Optimizer {
    pub const fn gradient_descent(learning_rate: f64) -> Optimizer {
        Optimizer::GradientDescent(GradientDescent { learning_rate })
    }

    pub const fn default_gradient_descent() -> Optimizer {
        Optimizer::GradientDescent(GradientDescent::default())
    }

    pub fn default_adam(layers: &Vec<Layer>) -> Optimizer {
        Optimizer::Adam(Adam::default(layers))
    }
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimizer::GradientDescent(opt) => write!(f, "{:?}", opt),
            Optimizer::Adam(opt) => write!(f, "{:?}", opt),
        }
    }
}
