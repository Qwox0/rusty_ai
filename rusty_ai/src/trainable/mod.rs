mod builder;

use crate::{gradient::aliases::OutputGradient, optimizer::IsOptimizer, prelude::*};
pub use builder::TrainableNeuralNetworkBuilder;

#[derive(Debug)]
pub struct TrainableNeuralNetwork<const IN: usize, const OUT: usize> {
    network: NeuralNetwork<IN, OUT>,
    gradient: Gradient,
    retain_gradient: bool,
    optimizer: Optimizer,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> TrainableNeuralNetwork<IN, OUT> {
    pub fn get_network(&self) -> &NeuralNetwork<IN, OUT> {
        &self.network
    }

    fn new(
        network: NeuralNetwork<IN, OUT>,
        optimizer: Optimizer,
        retain_gradient: bool,
        clip_gradient_norm: Option<ClipGradientNorm>,
    ) -> Self {
        let gradient = network.init_zero_gradient();
        Self { network, gradient, retain_gradient, optimizer, clip_gradient_norm }
    }

    /// calculates the outputs and derivatives of all layers
    pub fn verbose_propagate(&self, input: &[f64; IN]) -> VerbosePropagation {
        let layer_count = self.network.get_layers().len();
        let mut outputs = Vec::with_capacity(layer_count + 1);
        outputs.push(input.to_vec());

        for layer in self.network.iter_layers() {
            let input = outputs.last().expect("last element must exists");
            let output = layer.propagate(input);
            outputs.push(output);
        }
        VerbosePropagation::new(outputs)
    }

    /// Propagate a [`VerbosePropagation`] Result backwards through the Neural
    /// Network. This modifies the internal [`Gradient`].
    /// # Math
    ///    L-1                   L
    /// o_(L-1)_0
    ///                      z_0 -> o_L_0
    /// o_(L-1)_1    w_ij                    C
    ///                      z_1 -> o_L_1
    /// o_(L-1)_2
    ///         j              i        i
    /// n_(L-1) = 3           n_L = 2
    ///
    /// L: current Layer with n_L Neurons called L_1, L_2, ..., L_n
    /// L-1: previous Layer with n_(L-1) Neurons
    /// o_L_i: output of Neuron L_i
    /// e_i: expected output of Neuron L_i
    /// Cost: C = 0.5 * ∑ (o_L_i - e_i)^2 from i = 1 to n_L
    /// -> dC/do_L_i = o_L_i - e_i
    ///
    /// f: activation function
    /// activation: o_L_i = f(z_i)
    /// -> do_L_i/dz_i = f'(z_i)
    ///
    /// -> dC/dz_i = dC/do_L_i * do_L_i/dz_i = (o_L_i - e_i) * f'(z_i)
    ///
    /// w_ij: weight of connection from (L-1)_j to L_i
    /// b_L: bias of Layer L
    /// weighted sum: z_i = b_L + ∑ w_ij * o_(L-1)_j from j = 1 to n_(L-1)
    /// -> dz_i/dw_ij      = o_(L-1)_j
    /// -> dz_i/do_(L-1)_j = w_ij
    /// -> dz_i/dw_ij      = 1
    ///
    ///
    /// dC/dw_ij      = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i) *  o_(L-1)_j
    /// dC/do_(L-1)_j = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i) *       w_ij
    /// dC/db_L       = dC/do_L_i     * do_L_i/dz_i * dz_i/dw_ij
    ///               = (o_L_i - e_i) *     f'(z_i)
    pub fn backpropagation(
        &mut self,
        verbose_prop: VerbosePropagation,
        expected_output: &[f64; OUT],
    ) {
        let network_output = verbose_prop.outputs.last().expect("There is an output layer");

        // gradient of the cost function with respect to the neuron output of the last layer.
        let mut output_gradient = self.output_gradient(network_output, expected_output);
        let in_out_pairs = verbose_prop.outputs.array_windows::<2>();

        for ((layer, gradient), [input, output]) in
            self.network.iter_layers().zip(self.gradient.iter_mut_layers()).zip(in_out_pairs).rev()
        {
            output_gradient = layer.backpropagate(input, output, output_gradient, gradient);
        }
    }

    /// gradient of the loss with respect to the last neuron activations
    #[inline]
    fn output_gradient(
        &self,
        last_output: &Vec<f64>,
        expected_output: &[f64; OUT],
    ) -> OutputGradient {
        self.network.error_function.gradient(last_output, expected_output)
    }
}

impl<const IN: usize, const OUT: usize> Trainer<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    type Trainee = NeuralNetwork<IN, OUT>;

    fn get_trainee(&self) -> &Self::Trainee {
        &self.network
    }

    fn get_gradient_mut(&mut self) -> &mut Gradient {
        &mut self.gradient
    }

    fn calc_gradient<'a>(&mut self, batch: impl IntoIterator<Item = &'a Pair<IN, OUT>>) {
        for (input, expected_output) in batch.into_iter().map(Into::into) {
            let out = self.verbose_propagate(input);
            self.backpropagation(out, expected_output);
        }
    }

    fn optimize_trainee(&mut self) {
        self.optimizer.optimize_weights(&mut self.network, &self.gradient);
    }

    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>) {
        if !self.retain_gradient {
            self.set_zero_gradient();
        }

        self.calc_gradient(data_pairs);

        if let Some(clip_gradient_norm) = self.clip_gradient_norm {
            self.clip_gradient(clip_gradient_norm);
        } else {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        self.optimize_trainee()
    }
}

impl<const IN: usize, const OUT: usize> Propagator<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.network.propagate(input)
    }

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT> {
        self.network.test_propagate(data_pairs)
    }
}

impl<const IN: usize, const OUT: usize> std::fmt::Display for TrainableNeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.network)?;
        write!(f, "Optimizer: {}", self.optimizer)
    }
}

#[cfg(test)]
mod benches;
