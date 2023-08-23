mod builder;

use crate::prelude::*;
pub use builder::*;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NNTrainer<const IN: usize, const OUT: usize, L, O>
where
    L: LossFunction<OUT>,
    O: Optimizer,
{
    network: NeuralNetwork<IN, OUT>,
    gradient: Gradient,
    loss_function: L,
    retain_gradient: bool,
    optimizer: O,
    clip_gradient_norm: Option<ClipGradientNorm>,
}

impl<const IN: usize, const OUT: usize, L, EO, O> NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
{
    fn new(
        network: NeuralNetwork<IN, OUT>,
        loss_function: L,
        optimizer: O,
        retain_gradient: bool,
        clip_gradient_norm: Option<ClipGradientNorm>,
    ) -> Self {
        let gradient = network.init_zero_gradient();
        Self { network, gradient, loss_function, retain_gradient, optimizer, clip_gradient_norm }
    }

    #[inline]
    pub fn get_loss_function(&self) -> &L {
        &self.loss_function
    }

    /// calculates the output of every layer
    pub fn verbose_propagate(&self, input: &[f64; IN]) -> VerbosePropagation<OUT> {
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

    pub fn calculate_loss(&self, res: &PropagationResult<OUT>, expected_output: &EO) -> f64 {
        self.loss_function.propagate_arr(&res.0, expected_output)
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
    pub fn backpropagation(&mut self, verbose_prop: VerbosePropagation<OUT>, expected_output: &EO) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let mut output_gradient = self.loss_function.backpropagate(&verbose_prop, expected_output);

        let layer_iter = self.network.iter_layers();
        let grad_iter = self.gradient.iter_mut_layers();
        let in_out_pairs = verbose_prop.iter_layers();

        for ((layer, gradient), prop) in layer_iter.zip(grad_iter).zip(in_out_pairs).rev() {
            let LayerPropagation { input, output } = prop;
            output_gradient = layer.backpropagate(input, output, output_gradient, gradient);
        }
    }

    pub fn test<'a>(
        &self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, EO>>,
    ) -> TestsResult<OUT>
    where
        EO: 'a,
    {
        self.network.test(&self.loss_function, data_pairs)
    }

    pub fn get_trainee(&self) -> &NeuralNetwork<IN, OUT> {
        &self.network
    }

    pub fn get_gradient_mut(&mut self) -> &mut Gradient {
        &mut self.gradient
    }

    /// this calls `set_zero_gradient` based on the `self.retain_gradient` option.
    pub fn option_set_zero_gradient(&mut self) {
        if !self.retain_gradient {
            self.set_zero_gradient();
        }
    }

    pub fn set_zero_gradient(&mut self) {
        self.get_gradient_mut().set_zero()
    }

    pub fn calc_gradient<'a>(&mut self, batch: impl IntoIterator<Item = &'a Pair<IN, EO>>)
    where EO: 'a {
        for (input, expected_output) in batch.into_iter().map(Into::into) {
            let out = self.verbose_propagate(input);
            self.backpropagation(out, expected_output);
        }
    }

    pub fn clip_gradient(&mut self, clip_gradient_norm: ClipGradientNorm) {
        clip_gradient_norm.clip_gradient_pytorch(self.get_gradient_mut());
        //clip_gradient_norm.clip_gradient_pytorch_device(&mut self.gradient);
    }

    pub fn optimize_trainee(&mut self) {
        self.optimizer.optimize_weights(&mut self.network, &self.gradient);
    }

    /// Trains the neural network for one step/generation. Uses a small data set `data_pairs` to
    /// find an approximation for the weights gradient. The neural network's Optimizer changes the
    /// weights by using the calculated gradient.
    pub fn training_step<'a>(&mut self, data_pairs: impl IntoIterator<Item = &'a Pair<IN, EO>>)
    where EO: 'a {
        self.option_set_zero_gradient();

        self.calc_gradient(data_pairs);

        if let Some(clip_gradient_norm) = self.clip_gradient_norm {
            self.clip_gradient(clip_gradient_norm);
        } else {
            #[cfg(debug_assertions)]
            eprintln!("WARN: It is recommended to clip the gradient")
        }

        self.optimize_trainee()
    }

    /// Trains the `Self::Trainee` for `epoch_count` epochs. Each epoch the entire `training_data`
    /// is used to calculated the gradient approximation.
    pub fn full_train(
        &mut self,
        training_data: &PairList<IN, EO>,
        epoch_count: usize,
        mut callback: impl FnMut(usize, &NeuralNetwork<IN, OUT>),
    ) {
        for epoch in 1..=epoch_count {
            self.training_step(training_data.iter());
            callback(epoch, self.get_trainee());
        }
    }

    pub fn train(
        &mut self,
        training_data: &PairList<IN, EO>,
        training_amount: usize,
        epoch_count: usize,
        mut callback: impl FnMut(usize, &NeuralNetwork<IN, OUT>),
    ) {
        let mut rng = rand::thread_rng();
        for epoch in 1..=epoch_count {
            let training_data = training_data.choose_multiple(&mut rng, training_amount);
            self.training_step(training_data);
            callback(epoch, self.get_trainee());
        }
    }
}

impl<const IN: usize, const OUT: usize, EO, L, O> Propagator<IN, OUT> for NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
{
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.network.propagate(input)
    }

    /*
    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT> {
        self.network.test_propagate(data_pairs)
    }
    */
}

impl<const IN: usize, const OUT: usize, L, EO, O> Display for NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.network)?;
        write!(f, "Loss Function: {}, Optimizer: {}", self.loss_function, self.optimizer)
    }
}

#[cfg(test)]
mod benches;
