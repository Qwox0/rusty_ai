#![allow(incomplete_features)]
#![feature(array_windows)]
#![feature(test)]
#![feature(type_changing_struct_update)]
#![feature(iter_array_chunks)]
#![feature(portable_simd)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(associated_type_defaults)]
#![feature(exact_size_is_empty)]
#![feature(generic_const_exprs)]
#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod clip_gradient_norm;
pub mod initializer;
pub mod loss_function;
pub mod nn;
mod norm;
pub mod optimizer;
mod pair;
pub mod reexport;
pub mod test_result;
pub mod test_util;
pub mod trainer;
mod util;

pub use const_tensor::{self, Element, Float, Multidimensional, MultidimensionalOwned, Num};
pub use nn::{NNBuilder, NN};
pub use norm::Norm;
pub use optimizer::Optimizer;
pub use pair::Pair;
pub use reexport::half::{bf16, f16};

/// # `rusty_ai` prelude
///
/// includes everything
pub mod prelude {
    pub use crate::{
        clip_gradient_norm::*, const_tensor::*, initializer::*, loss_function::*,
        nn::builder::markers::*, optimizer::*, *,
    };
}

// Ai Training steps
// n: number of data points
// 1. Propagation function (every Artificial neurons) weighted sum: net_j = ∑ x_i*w_(ij) from i = 1
//    to m (m: number of neurons
// in layer j-1) 2. Activation function (every Artificial neurons)
//      ReLU: o_j = max(0, net_j)                               (for hidden
// layers)      sigmoid: o_j = 1/(1+e^(-net_j)) = e^net_j/(e^net_j + 1) (for
// output layer) 3. Loss/Cost/Error function (on result vector)
//      Mean squarred error: E = 0.5 * ∑ (o_i - t_i)^2 from i = 1 to n
//
//      o_i: components of the result vector
//      t_i: components of the target vector (correct solution, given data)
// 4. Backpropagation gradient of Error = ∇ E -> dE/dw_(ij)   =
// dE/do_j*do_j/dnet_j*dnet_j/w_(ij)      with the given examples:
//          dE/do_j         = o_j - t_j
//                            {0 when net_j<0
//          do_j/dnet_j     = {1 when net_j>0
// (ReLU)                            {? when net_j=0
//          do_j/dnet_j     = e^(-net_j)/(1+e^(-net_j))^2 =
// e^net_j/(1+e^net_j)^2   (sigmoid)          dnet_j/w_(ij)   = x_i
//
//          => dE/dw_(ij)   = (o_j - t_j) * df/dw_(ij)(∑ x_k*w_(kj) from k = 1
// to n) * x_i 5. Optimizer
//      improve weights:
//      w_(ij) = w_(ij) - a * dE/dw_(ij)
//
//      a: learning rate
//
//      ADAM
//      Stochastic gradient descent
//
// Example:
//  w = (w1 w2)^T
//  y = w1 + w2*x
//  dataset: (x1, x2, ..., xn), (y1, y2, ..., yn)
//  activation function: f(x) = x
//
//  E(w) = ∑ E_i(w) = ∑ (y - yi)^2 = ∑ (w1 + w2*xi - yi)^2
//  (all sums from i=1 to n)
//
//  Iterate:
//  w := w - a*∇ E(w) = w - a * [d/dw1(w1 + w2*xi - yi)^2 d/dw2(w1 + w2*xi - yi)^2]^T
//     = w - a * [2(w1 + w2*xi - yi) 2xi(w1 + w2*xi - yi)]^T
