mod adam;
mod gradient_descent;

pub use adam::Adam;
pub use gradient_descent::GradientDescent;

use crate::{gradient::Gradient, layer::Layer, neural_network::NeuralNetwork};
use enum_dispatch::enum_dispatch;

pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

#[enum_dispatch]
pub(crate) trait IsOptimizer {
    fn optimize_weights<const IN: usize, const OUT: usize>(
        &mut self,
        nn: &mut NeuralNetwork<IN, OUT>,
        gradient: &Gradient,
    );
    #[allow(unused)]
    fn init_with_layers(&mut self, layers: &Vec<Layer>) {}
}

#[derive(Debug)]
#[enum_dispatch(IsOptimizer)]
pub enum Optimizer {
    GradientDescent(GradientDescent),
    Adam(Adam),
}

impl Optimizer {
    pub fn gradient_descent(learning_rate: f64) -> Optimizer {
        Optimizer::GradientDescent(GradientDescent { learning_rate })
    }
    pub fn default_gradient_descent() -> Optimizer {
        Optimizer::GradientDescent(GradientDescent::default())
    }

    pub fn default_adam() -> Optimizer {
        Optimizer::Adam(Adam::default())
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
