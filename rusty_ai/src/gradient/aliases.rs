use crate::bias::LayerBias;
use matrix::Matrix;

//    L-1                   L
// o_(L-1)_0
//                      z_0 -> o_L_0
// o_(L-1)_1    w_ij                    C
//                      z_1 -> o_L_1
// o_(L-1)_2
//         j              i        i
// n_(L-1) = 3           n_L = 2
//
// L: current Layer with n_L Neurons called L_1, L_2, ..., L_n
// L-1: previous Layer with n_(L-1) Neurons
// o_L_i: output of Neuron L_i
// e_i: expected output of Neuron L_i
// Cost: C = 0.5 * ∑ (o_L_i - e_i)^2 from i = 1 to n_L
// -> dC/do_L_i = o_L_i - e_i
//
// f: activation function
// activation: o_L_i = f(z_i)
// -> do_L_i/dz_i = f'(z_i)
//
// -> dC/dz_i = dC/do_L_i * do_L_i/dz_i = (o_L_i - e_i) * f'(z_i)
//
// w_ij: weight of connection from (L-1)_j to L_i
// b_L: bias of Layer L
// weighted sum: z_i = b_L + ∑ w_ij * o_(L-1)_j from j = 1 to n_(L-1)
// -> dz_i/dw_ij      = o_(L-1)_j
// -> dz_i/do_(L-1)_j = w_ij
// -> dz_i/dw_ij      = 1
//
//
// dC/dw_ij      = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
//               = (o_L_i - e_i) *     f'(z_i) *  o_(L-1)_j
// dC/do_(L-1)_j = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
//               = (o_L_i - e_i) *     f'(z_i) *       w_ij
// dC/db_L       = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
//               = (o_L_i - e_i) *     f'(z_i)

/// derivatives of the total cost with respect to the neuron activations
pub type OutputGradient = Vec<f64>;

/// derivatives of the total cost with respect to the weighted sums
pub type WeightedSumGradient = Vec<f64>;

/// derivatives of the total cost with respect to the incoming weights
pub type WeightGradient = Matrix<f64>;
/// derivatives of the total cost with respect to the previous neuron
/// activations
pub type InputGradient = Vec<f64>;
/// derivatives of the total cost with respect to the bias/biases
pub type BiasGradient = LayerBias;
