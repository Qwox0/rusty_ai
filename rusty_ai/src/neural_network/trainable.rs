use crate::prelude::*;
use crate::traits::IterLayerParams;
use crate::util::Norm;
use crate::{
    optimizer::IsOptimizer,
    traits::{Propagator, Trainable},
};

#[derive(Debug)]
pub struct ClipGradientNorm {
    pub norm_type: Norm,
    pub max_norm: f64,
}

#[derive(Debug)]
pub struct TrainableNeuralNetwork<const IN: usize, const OUT: usize> {
    network: NeuralNetwork<IN, OUT>,
    gradient: Gradient,
    retain_gradient: bool,
    optimizer: Optimizer,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize> TrainableNeuralNetwork<IN, OUT> {
    pub(crate) fn new(
        network: NeuralNetwork<IN, OUT>,
        optimizer: Optimizer,
        retain_gradient: bool,
        clip_gradient_norm: Option<ClipGradientNorm>,
    ) -> Self {
        let gradient = network.init_zero_gradient();
        Self {
            network,
            gradient,
            retain_gradient,
            optimizer,
            clip_gradient_norm,
        }
    }

    /// calculates the outputs and derivatives of all layers
    fn verbose_propagate(&self, input: &[f64; IN]) -> VerbosePropagation {
        let layer_count = self.network.get_layers().len();
        let mut outputs = Vec::with_capacity(layer_count + 1);
        outputs.push(input.to_vec());
        let mut derivatives = Vec::with_capacity(layer_count);

        for layer in self.network.iter_layers() {
            let input = outputs.last().expect("last element must exists");
            let (output, derivative) = layer.training_calculate(input);
            outputs.push(output);
            derivatives.push(derivative);
        }
        VerbosePropagation::new(outputs, derivatives)
    }

    /// Propagate a [`VerbosePropagation`] Result backwards through the Neural Network. This
    /// modifies the internal [`Gradient`].
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
        let mut outputs = verbose_prop.outputs;
        let last_output = outputs.pop().expect("There is an output layer");
        let expected_output = expected_output.to_vec();

        // derivatives of the cost function with respect to the output of the neurons in the last layer.
        let last_output_gradient = self
            .network
            .error_function
            .gradient(last_output, expected_output); // dC/do_L_i; i = last
        let inputs_rev = outputs.into_iter().rev();

        self.network
            .layers
            .iter()
            .zip(self.gradient.iter_mut_layers())
            .zip(verbose_prop.derivatives)
            .rev()
            .zip(inputs_rev)
            .fold(
                last_output_gradient,
                |current_output_gradient, (((layer, gradient), derivative_output), input)| {
                    layer.backpropagation2(
                        derivative_output,
                        input,
                        current_output_gradient,
                        gradient,
                    )
                },
            );
    }

    fn optimize(&mut self) {
        self.optimizer
            .optimize_weights(&mut self.network, &self.gradient);
        self.network.increment_generation();
    }
}

impl<const IN: usize, const OUT: usize> Trainable<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    type Trainee = NeuralNetwork<IN, OUT>;

    fn train(
        &mut self,
        training_data: &PairList<IN, OUT>,
        training_amount: usize,
        epoch_count: usize,
        mut callback: impl FnMut(usize, &Self::Trainee),
    ) {
        let mut rng = rand::thread_rng();
        for epoch in 1..=epoch_count {
            let training_data = training_data.choose_multiple(&mut rng, training_amount);
            self.training_step(training_data);
            callback(epoch, &self.network);
        }
    }

    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>) {
        if !self.retain_gradient {
            self.gradient = self.network.init_zero_gradient();
        }

        for (input, expected_output) in data_pairs.into_iter().map(Into::into) {
            self.backpropagation(self.verbose_propagate(input), expected_output);
        }

        if let Some(clip_gradient_norm) = &self.clip_gradient_norm {
            let iter = self.gradient.iter_parameters().cloned();
            let norm = clip_gradient_norm.norm_type.calculate(iter);
            if norm > clip_gradient_norm.max_norm {
                let clip_factor = clip_gradient_norm.max_norm / (norm + 1e-6); // 1e-6 copied from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
                self.gradient
                    .iter_mut_parameters()
                    .for_each(|x| *x *= clip_factor);
            }
        } else {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        self.optimize()
    }

    /*
    fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a DataPair<IN, OUT>>) {
        let mut data_count = 0;
        let mut gradient = self.network.init_zero_gradient();
        for (input, expected_output) in data_pairs.into_iter().map(Into::into) {
            self.network.backpropagation(
                self.verbose_propagate(input),
                expected_output,
                &mut gradient,
            );
            data_count += 1;
        }
        /*
        if data_count < 5 {
            eprintln!("WARN: Small training sets result in inaccurate gradients which might cause exploding weight values!")
        }
        */

        // average of all gradients
        gradient.normalize(data_count);

        self.optimize(gradient);
    }
    */
}

impl<const IN: usize, const OUT: usize> Propagator<IN, OUT> for TrainableNeuralNetwork<IN, OUT> {
    fn propagate(&self, input: &[f64; IN]) -> crate::results::PropagationResult<OUT> {
        self.network.propagate(input)
    }

    fn propagate_many(
        &self,
        input_list: &Vec<[f64; IN]>,
    ) -> Vec<crate::results::PropagationResult<OUT>> {
        self.network.propagate_many(input_list)
    }

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> crate::results::TestsResult<OUT> {
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
