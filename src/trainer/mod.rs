//! Trainer module.

use crate::{clip_gradient_norm::ClipGradientNorm, training::Training, *};
use data::Pair;
use loss_function::LossFunction;
use rand::seq::IteratorRandom;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, fmt::Display, iter::Map};

mod builder;
pub use builder::{markers, NNTrainerBuilder};

/// Trainer for a [`NeuralNetwork`].
///
/// Can be constructed using a [`NNTrainerBuilder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NNTrainer<const IN: usize, const OUT: usize, L, O> {
    network: NeuralNetwork<IN, OUT>,
    gradient: Gradient,
    loss_function: L,
    retain_gradient: bool,
    optimizer: O,
    clip_gradient_norm: Option<ClipGradientNorm>,
    // training_threads: usize,
}

impl<const IN: usize, const OUT: usize, L, O> NNTrainer<IN, OUT, L, O> {
    /// Returns a reference to the underlying [`NeuralNetwork`].
    pub fn get_network(&self) -> &NeuralNetwork<IN, OUT> {
        &self.network
    }

    /// Returns a reference to the internal [`Gradient`].
    pub fn get_gradient(&self) -> &Gradient {
        &self.gradient
    }

    /// Converts `self` into the underlying [`NeuralNetwork`]. This can be used after the training
    /// is finished.
    pub fn into_nn(self) -> NeuralNetwork<IN, OUT> {
        self.network
    }

    /// The [`Gradient`] must have the correct dimensions.
    ///
    /// It is recommended to create the Gradient with `NeuralNetwork::init_zero_gradient`.
    pub fn unchecked_set_gradient(&mut self, gradient: Gradient) {
        self.gradient = gradient;
    }

    /// The [`Gradient`] must have the correct dimensions.
    ///
    /// It is recommended to create the Gradient with `NeuralNetwork::init_zero_gradient`.
    pub fn unchecked_add_gradient(&mut self, gradient: Gradient) {
        self.gradient += gradient;
    }

    /// Propagates an [`Input`] through the underlying neural network and returns its output.
    #[inline]
    pub fn propagate(&self, input: &Input<IN>) -> [f64; OUT] {
        self.network.propagate(input)
    }

    /// Iterates over a `batch` of inputs and returns an [`Iterator`] over the outputs.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    ///
    /// If you also want to calculate losses use `test` or `prop_with_test`.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn propagate_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(&'a Input<IN>) -> [f64; OUT]>
    where
        B: IntoIterator<Item = &'a Input<IN>>,
    {
        self.network.propagate_batch(batch)
    }

    /// Propagates an [`Input`] through the underlying neural network and returns the input and the
    /// outputs of every layer.
    ///
    /// If only the final output is needed, use `propagate` instead.
    ///
    /// This is used internally during training.
    #[inline]
    pub fn verbose_propagate(&self, input: &Input<IN>) -> VerbosePropagation<OUT> {
        self.network.verbose_propagate(input)
    }

    /// Sets every parameter of the interal [`Gradient`] to `0.0`.
    #[inline]
    pub fn set_zero_gradient(&mut self) {
        self.gradient.set_zero()
    }

    /// Sets every parameter of the interal [`Gradient`] to `0.0` if `self.retain_gradient` is
    /// `false`, otherwise this does nothing.
    ///
    /// If you always want to reset the [`Gradient`] use `set_zero_gradient` instead.
    pub fn maybe_set_zero_gradient(&mut self) {
        if !self.retain_gradient {
            self.set_zero_gradient();
        }
    }

    /// Clips the internal [`Gradient`] based on `self.clip_gradient_norm`.
    ///
    /// If `self.clip_gradient_norm` is [`None`], this does nothing.
    pub fn clip_gradient(&mut self) {
        if let Some(clip_gradient_norm) = self.clip_gradient_norm {
            clip_gradient_norm.clip_gradient_pytorch(&mut self.gradient);
            //clip_gradient_norm.clip_gradient_pytorch_device(&mut self.gradient);
        }
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> NNTrainer<IN, OUT, L, O>
where L: LossFunction<OUT, ExpectedOutput = EO>
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

    /// Gets the [`LossFunction`] used during training.
    #[inline]
    pub fn get_loss_function(&self) -> &L {
        &self.loss_function
    }

    /// Propagate a [`VerbosePropagation`] Result backwards through the Neural
    /// Network. This modifies the internal [`Gradient`].
    ///
    /// # Math
    ///
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
    pub fn backpropagate(&mut self, verbose_prop: &VerbosePropagation<OUT>, expected_output: &EO) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let output_gradient = self.loss_function.backpropagate(verbose_prop, expected_output);
        self.network.backpropagate(verbose_prop, output_gradient, &mut self.gradient);
    }

    /// Like `backpropagate` but modifies the `gradient` parameter instead of the internal gradient
    pub fn backpropagate_into(
        &self,
        verbose_prop: &VerbosePropagation<OUT>,
        expected_output: &EO,
        gradient: &mut Gradient,
    ) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let output_gradient = self.loss_function.backpropagate(verbose_prop, expected_output);
        self.network.backpropagate(verbose_prop, output_gradient, gradient);
    }

    /// To test a batch of multiple pairs use `test_batch`.
    #[inline]
    pub fn test(&self, input: &Input<IN>, expected_output: &EO) -> ([f64; OUT], f64) {
        self.network.test(input, expected_output, &self.loss_function)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// outputs and the losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn test_batch<'a, B>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(&'a Pair<IN, EO>) -> ([f64; OUT], f64)>
    where
        B: IntoIterator<Item = &'a Pair<IN, EO>>,
        EO: 'a,
    {
        batch.into_iter().map(|(input, eo)| self.test(input, eo))
    }
}

impl<const IN: usize, const OUT: usize, L, O> NNTrainer<IN, OUT, L, O>
where O: Optimizer
{
    /// Uses the internal [`Optimizer`] and [`Gradient`] to optimize the internal
    /// [`NeuralNetwork`] once.
    #[inline]
    pub fn optimize_trainee(&mut self) {
        self.optimizer.optimize(&mut self.network, &self.gradient);
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO>,
    O: Optimizer,
{
    /// Trains the internal [`NeuralNetwork`] lazily.
    #[inline]
    //pub fn train<'a, B>(&'a mut self, batch: B) -> Training<Self, B::IntoIter>
    //where
    //    B: IntoIterator<Item = &'a Pair<IN, EO>>,
    //    EO: 'a,
    //{
    //    Training::new(self, batch.into_iter())
    //}
    pub fn train<'a>(&'a mut self, batch: &'a [Pair<IN, EO>]) -> Training<'a, Self, Pair<IN, EO>>
    where EO: 'a {
        Training::new(self, batch.as_ref())
    }

    /// creates a sample [`NNTrainer`].
    ///
    /// This is probably only useful for testing.
    pub fn default<V>() -> Self
    where
        L: Default,
        V: OptimizerValues<Optimizer = O> + Default,
    {
        NeuralNetwork::default()
            .to_trainer()
            .loss_function(L::default())
            .optimizer(V::default())
            .build()
    }
}

impl<const IN: usize, const OUT: usize, L, EO, O> Display for NNTrainer<IN, OUT, L, O>
where
    L: LossFunction<OUT, ExpectedOutput = EO> + Display,
    O: Optimizer + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.network)?;
        write!(f, "Loss Function: {}, Optimizer: {}", self.loss_function, self.optimizer)
    }
}

#[cfg(test)]
mod benches;

#[cfg(test)]
mod seeded_tests {
    use crate::{
        optimizer::sgd::SGD, prelude::SquaredError, ActivationFn, BuildLayer, Initializer, Input,
        NNBuilder, Norm, ParamsIter,
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_propagation() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let ai = NNBuilder::default()
            .rng(&mut rng)
            .input::<2>()
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::ReLU)
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::ReLU)
            .layer(3, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::Sigmoid)
            .build::<3>();

        let out = ai.propagate(&Input::new(Box::new(rng.gen())));

        let expected = [0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(out, expected, "incorrect output");
    }

    #[test]
    fn test_gradient() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut ai = NNBuilder::default()
            .rng(&mut rng)
            .input::<2>()
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::LeakyReLU { leak_rate: 0.1 })
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::Identity)
            .layer(3, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::Sigmoid)
            .build::<3>()
            .to_trainer()
            .loss_function(SquaredError)
            .optimizer(SGD::default())
            .retain_gradient(true)
            .new_clip_gradient_norm(5.0, Norm::Two)
            .build();

        let pairs = (0..5)
            .map(|_| Input::new(Box::new(rng.gen())))
            .map(|input| {
                let sum = input.iter().sum();
                let prod = input.iter().product();
                (input, [sum, prod, 0.0])
            })
            .collect::<Vec<_>>();
        ai.train(&pairs).execute();

        let gradient = ai.get_gradient().iter().copied();
        #[rustfmt::skip]
        let expected = &[-0.31527687134612725, -0.2367818186318013, 0.020418606049140496, 0.015550533247454118, -0.001738594474681532, -0.004567137662788988, 0.033859035247529076, 0.02977764834407858, 0.009605083538032784, 0.006826476555722042, -0.6146889718563872, 0.040329523770678166, -0.0050435074713498255, 0.07884882135388332, 0.01821395483674612, -0.29603217316507463, 0.03346627048986953, -0.05878503376051679, -0.2158320807897315, 0.026983231976989056, -0.24554656005324604, 0.02887071882597527, -0.05843956975876532, -0.17243502863504168, 0.022467020003968326, 0.1311511999366582, -0.014868359886407069, 0.02611204347681894, 0.09796223173142266, -0.01217871097874241, -0.054034033529765414, 0.005815532263211411, -0.008911633440899904, -0.034708137931585205, 0.0043542213467064154, 0.2911033451780113, -0.033617414595422807, 0.06373660842764983, 0.21011649849556374, -0.026765687357284716, -0.6362499755696547, -0.5444227081050653, 0.288719083038049, -0.09632461875232448, 0.6412639438519863, -0.2336931253665935, -0.005400196361134401, 0.34277389227803257, -0.01683276960395113, 0.18829397721654298, 0.22312222066614518, 0.0668809841607016, -0.21284993370400693, 0.004724430001813006, -0.06979054513916816, 0.6470015756088618, 0.15347488461571596, -0.7149338885041989, 0.04221164029892477, -0.30940480020839184, -0.4461655070529444, 0.4131071925533384, 1.1799836756082676];
        let err = gradient.zip(expected).map(|(p, e)| (p - e).abs()).sum::<f64>();
        assert!(err < 1e-15, "incorrect gradient elements (err: {})", err);
    }

    #[test]
    fn test_training() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut ai = NNBuilder::default()
            .rng(&mut rng)
            .input::<2>()
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::ReLU)
            .layer(5, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::ReLU)
            .layer(3, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .activation_function(ActivationFn::Sigmoid)
            .build::<3>()
            .to_trainer()
            .loss_function(SquaredError)
            .optimizer(SGD::default())
            .retain_gradient(true)
            .new_clip_gradient_norm(5.0, Norm::Two)
            .build();

        // nn params pre training
        let params = ai.get_network().iter().copied().collect::<Vec<_>>();
        #[rustfmt::skip]
        let expected = &[0.006887838447803829, 0.6393999304942023, 0.34936912393918684, -0.4047589840789866, -0.37941201236065963, -0.06972914538603359, 0.43380493311798884, 0.2808271748488419, -0.16417981196958276, -0.2391556648674174, 0.2108008059374471, -0.5508539013658884, 0.30609095651501483, 0.0010017747733542803, -0.2439626553503762, 0.1739790758995245, -0.35504329049611705, 0.2807057469930026, -0.021561872961492812, -0.2224985097439988, 0.18025297158732995, -0.3118176626548729, 0.26646269895835534, -0.4111905543260018, 0.07174135969857715, -0.3910179151410674, -0.14027757282776454, 0.39256214288992813, -0.1804116593475944, -0.06183204149127286, 0.30148591157620747, -0.07045111402421522, 0.15330561621693045, -0.05987140494810189, 0.16392997905786127, -0.41157175802213586, 0.06448666319062674, 0.3549482907502232, -0.1752947400236416, 0.17664346553608495, 0.4130563079306305, 0.12362639119103341, -0.4340562639542757, -0.09883618080186729, -0.05709696039076012, 0.3577843982370516, 0.1972113234741723, -0.2053210678418987, -0.03384982362548067, -0.32891932430635185, -0.26690036384241284, -0.24283061456061486, 0.23016935417459677, 0.23254520394702988, 0.3651839637543794, -0.310479259746818, -0.3017997213731933, 0.08646500039777222, -0.17584424522752867, 0.29123399909249675, 0.06853079258143152, -0.3543537884722492, 0.2413959457728258];
        assert_eq!(&params, expected, "incorrect seed");

        let input = Input::new(Box::new(rng.gen()));
        let eo = rng.gen();
        let pair = (input, eo);

        // propagation pre training
        let prop_out = ai.propagate(&pair.0);
        #[rustfmt::skip]
        let expected = &[0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(&prop_out, expected, "incorrect propagation (pre training)");

        // do training
        //let res = ai.train([&pair]).execute();
        let pairs = [pair.clone()];
        let mut res_iter = ai.train(&pairs).losses();
        let res = res_iter.next().unwrap();
        assert!(res_iter.count() == 0);

        // gradient
        let gradient = ai.get_gradient().iter().copied();
        #[rustfmt::skip]
        let expected = &[-0.003572457096879446, -0.0002794669624026194, 0.0, 0.0, 0.0, 0.0, 0.0004427478746224825, 3.463537847355607e-5, 0.0, 0.0, -0.0038284717751311454, 0.0, 0.0, 0.0004744767244292752, 0.0, -0.00580726500680219, 0.0, 0.0, -0.009380821731006066, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022005357571517076, 0.0, 0.0, 0.0, 0.0, 0.008226304898399563, 0.0, 0.0, 0.0, 0.0, 0.06395275520239677, 0.0, 0.0, 0.0, 0.0, -0.013618442087793855, 0.0, 0.0, 0.0, 0.0, 0.018289699457633535, 0.1421873716797208, -0.030278140178756956];

        let err = gradient.zip(expected).map(|(p, e)| (p - e).abs()).sum::<f64>();
        assert!(err < 1e-15, "incorrect gradient elements (err: {})", err);

        // training output
        let expected = &[0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(&res.0, expected, "incorrect output");

        // training loss
        assert_eq!(res.1, 0.09546645303826229, "incorrect loss");

        // propagation post training
        let prop_out = ai.propagate(&pair.0);
        #[rustfmt::skip]
        let expected = &[0.5570843934685307, 0.38315307990347475, 0.525483857537024];
        assert_eq!(&prop_out, expected, "incorrect propagation (post training)");

        #[rustfmt::skip]
        const TRAINED_PARAMS: &[f64] = &[0.006923563018772624, 0.6394027251638263, 0.34936912393918684, -0.4047589840789866, -0.37941201236065963, -0.06972914538603359, 0.4338005056392426, 0.2808268284950572, -0.16417981196958276, -0.2391556648674174, 0.2108390906551984, -0.5508539013658884, 0.30609095651501483, 0.0009970300061099876, -0.2439626553503762, 0.1740371485495925, -0.35504329049611705, 0.2807057469930026, -0.021468064744182752, -0.2224985097439988, 0.18025297158732995, -0.3118176626548729, 0.26646269895835534, -0.4111905543260018, 0.07174135969857715, -0.3910179151410674, -0.14027757282776454, 0.39256214288992813, -0.1804116593475944, -0.06183204149127286, 0.30148591157620747, -0.07045111402421522, 0.15330561621693045, -0.05987140494810189, 0.16392997905786127, -0.41157175802213586, 0.06448666319062674, 0.3549482907502232, -0.1752947400236416, 0.17664346553608495, 0.41327636150634567, 0.12362639119103341, -0.4340562639542757, -0.09883618080186729, -0.05709696039076012, 0.3577021351880676, 0.1972113234741723, -0.2053210678418987, -0.03384982362548067, -0.32891932430635185, -0.2675398913944368, -0.24283061456061486, 0.23016935417459677, 0.23254520394702988, 0.3651839637543794, -0.3103430753259401, -0.3017997213731933, 0.08646500039777222, -0.17584424522752867, 0.29123399909249675, 0.06834789558685518, -0.3557756621890464, 0.24169872717461338];

        let err = ai
            .get_network()
            .iter()
            .copied()
            .zip(TRAINED_PARAMS)
            .map(|(p, e)| (p - e).abs())
            .sum::<f64>();
        assert!(err < 1e-15, "incorrect trained parameters (err: {})", err);

        let trained_params = ai.get_network().iter().copied().collect::<Vec<_>>();
        assert_eq!(&trained_params, TRAINED_PARAMS, "incorrect trained parameters");
    }
}
