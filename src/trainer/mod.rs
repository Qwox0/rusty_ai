//! Trainer module.

use crate::{
    clip_gradient_norm::ClipGradientNorm,
    loss_function::LossFunction,
    nn::{GradComponent, Pair, TestResult},
    optimizer::Optimizer,
    Norm, NN,
};
use const_tensor::{tensor, Element, Float, Shape, Tensor};
use core::fmt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, iter::Map, sync::mpsc};

mod builder;
pub use builder::{markers, NNTrainerBuilder};

/// Trainer for a [`NeuralNetwork`].
///
/// Can be constructed using a [`NNTrainerBuilder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NNTrainer<X: Element, IN: Shape, OUT: Shape, L, O: Optimizer<X>, NN_: NN<X, IN, OUT>> {
    network: NN_,
    gradient: NN_::Grad,
    loss_function: L,
    retain_gradient: bool,
    optimizer: O,
    opt_state: NN_::OptState<O>,
    clip_gradient_norm: Option<ClipGradientNorm<X>>,
    // training_threads: usize,
}

impl<X, IN, OUT, L, EO, O, NN_> NNTrainer<X, IN, OUT, L, O, NN_>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: Optimizer<X>,
    NN_: NN<X, IN, OUT>,
{
    fn new(
        network: NN_,
        loss_function: L,
        optimizer: O,
        retain_gradient: bool,
        clip_gradient_norm: Option<ClipGradientNorm<X>>,
    ) -> Self {
        let gradient = network.init_zero_grad();
        let opt_state = network.init_opt_state();
        Self {
            network,
            gradient,
            loss_function,
            retain_gradient,
            optimizer,
            opt_state,
            clip_gradient_norm,
        }
    }

    /// Converts `self` into the underlying [`NeuralNetwork`]. This can be used after the training
    /// is finished.
    pub fn into_nn(self) -> NN_ {
        self.network
    }

    pub fn get_network(&self) -> &NN_ {
        &self.network
    }

    /// Propagates an [`Input`] through the underlying neural network and returns its output.
    #[inline]
    pub fn propagate(&self, input: &tensor<X, IN>) -> Tensor<X, OUT> {
        self.network.prop(input.to_owned())
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
    ) -> Map<B::IntoIter, impl FnMut(&'a tensor<X, IN>) -> Tensor<X, OUT>>
    where
        B: IntoIterator<Item = &'a tensor<X, IN>>,
    {
        self.network.prop_batch(batch)
    }

    /// Propagates a [`Tensor`] through the underlying neural network and returns the output
    /// [`Tensor`] and additional data which is required for backpropagation.
    ///
    /// If only the output is needed, use the normal `propagate` method instead.
    pub fn train_prop(
        &self,
        input: &impl ToOwned<Owned = Tensor<X, IN>>,
    ) -> (Tensor<X, OUT>, NN_::StoredData) {
        self.network.train_prop(input.to_owned())
    }

    /// Clips the gradient based on `self.clip_gradient_norm`.
    ///
    /// If `self.clip_gradient_norm` is [`None`], this does nothing.
    pub fn clip_gradient(&mut self)
    where X: Float {
        if let Some(clip_gradient_norm) = self.clip_gradient_norm {
            clip_gradient_norm.clip_gradient_pytorch(&mut self.gradient);
            //clip_gradient_norm.clip_gradient_pytorch_device(&mut self.gradient);
        }
    }

    pub fn backpropagate_inner(
        &mut self,
        output: Tensor<X, OUT>,
        expected_output: &EO,
        train_data: NN_::StoredData,
    ) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let output_gradient = self.loss_function.backpropagate(&output, expected_output);
        self.network.backprop_inplace(output_gradient, train_data, &mut self.gradient);
    }

    pub fn backpropagate_inplace(
        &self,
        output: Tensor<X, OUT>,
        expected_output: &EO,
        train_data: NN_::StoredData,
        gradient: &mut NN_::Grad,
    ) {
        // gradient of the cost function with respect to the neuron output of the last layer.
        let output_gradient = self.loss_function.backpropagate(&output, expected_output);
        self.network.backprop_inplace(output_gradient, train_data, gradient);
    }

    pub fn backpropagate(
        &self,
        output: Tensor<X, OUT>,
        expected_output: &EO,
        train_data: NN_::StoredData,
        mut gradient: NN_::Grad,
    ) -> NN_::Grad {
        self.backpropagate_inplace(output, expected_output, train_data, &mut gradient);
        gradient
    }

    /// To test a batch of multiple pairs use `test_batch`.
    #[inline]
    pub fn test<EO_: Borrow<EO>>(&self, pair: &Pair<X, IN, EO_>) -> TestResult<X, OUT> {
        self.network.test(pair, &self.loss_function)
    }

    /// Iterates over a `batch` of input-label-pairs and returns an [`Iterator`] over the network
    /// outputs and the losses.
    ///
    /// This [`Iterator`] must be consumed otherwise no calculations are done.
    #[must_use = "`Iterators` must be consumed to do work."]
    #[inline]
    pub fn test_batch<'a, B, EO_>(
        &'a self,
        batch: B,
    ) -> Map<B::IntoIter, impl FnMut(&'a Pair<X, IN, EO_>) -> TestResult<X, OUT>>
    where
        B: IntoIterator<Item = &'a Pair<X, IN, EO_>>,
        EO_: Borrow<EO> + 'a,
    {
        batch.into_iter().map(|p| self.test(p))
    }

    #[inline]
    pub fn optimize_trainee(&mut self) {
        self.network.optimize(&self.gradient, &self.optimizer, &mut self.opt_state);
    }
}

impl<X, IN, OUT, L, EO, O, NN_> NNTrainer<X, IN, OUT, L, O, NN_>
where
    X: Float,
    IN: Shape,
    OUT: Shape,
    NN_: NN<X, IN, OUT>,
    L: LossFunction<X, OUT, ExpectedOutput = EO>,
    O: Optimizer<X>,
{
    /// Trains the internal [`NN`].
    #[inline]
    pub fn train_single_thread<'a, EO_: Borrow<EO> + 'a>(
        &'a mut self,
        batch: impl IntoIterator<Item = &'a Pair<X, IN, EO_>>,
    ) {
        if !self.retain_gradient {
            self.gradient.set_zero();
        }
        for (input, eo) in batch.into_iter().map(Into::into) {
            let (out, data) = self.train_prop(input);
            self.backpropagate_inner(out, eo.borrow(), data);
        }
        self.clip_gradient();
        self.optimize_trainee();
    }

    /// Trains the internal [`NN`] lazily.
    #[inline]
    pub fn train_single_thread_output<'a, EO_: Borrow<EO> + 'a>(
        &'a mut self,
        batch: impl IntoIterator<Item = &'a Pair<X, IN, EO_>>,
    ) -> impl Iterator<Item = TrainOut<X, OUT>> {
        if !self.retain_gradient {
            self.gradient.set_zero();
        }
        let mut ret = Vec::new();
        for (input, eo) in batch.into_iter().map(Into::into) {
            let eo = eo.borrow();
            let (output, data) = self.network.train_prop(input.to_owned());
            let loss = self.loss_function.propagate(&output, eo);
            self.backpropagate_inner(output.clone(), eo, data);

            ret.push(TrainOut { output, loss });
        }
        self.clip_gradient();
        self.optimize_trainee();
        ret.into_iter()
    }

    /// Trains the internal [`NN`].
    #[inline]
    pub fn train_rayon<'a, EO_>(
        &'a mut self,
        batch: impl IntoParallelIterator<Item = &'a Pair<X, IN, EO_>>,
    ) where
        EO_: Borrow<EO> + 'a,
        NN_::Grad: Send + Sync,
        NN_::OptState<O>: Send + Sync,
    {
        if !self.retain_gradient {
            self.gradient.set_zero();
        }
        let grad = batch
            .into_par_iter()
            .fold(
                || self.network.init_zero_grad(),
                |grad, p| {
                    let (input, eo) = p.as_tuple();
                    let (out, data) = self.train_prop(input);
                    NNTrainer::backpropagate(self, out, eo.borrow(), data, grad)
                },
            )
            //.reduce(|| self.network.init_zero_grad(), |acc, grad| acc.add(&grad));
            .reduce(|| self.network.init_zero_grad(), GradComponent::add);
        self.gradient.add_mut(&grad);
        self.clip_gradient();
        self.optimize_trainee();
    }

    /// Trains the internal [`NN`].
    #[inline]
    pub fn train_rayon_output<'a, EO_>(
        &'a mut self,
        batch: impl IntoParallelIterator<Item = &'a Pair<X, IN, EO_>>,
    ) -> impl Iterator<Item = TrainOut<X, OUT>>
    where
        EO_: Borrow<EO> + 'a,
        IN: Send + Sync,
        OUT: Send + Sync,
        NN_::Grad: Send + Sync,
        NN_::OptState<O>: Send + Sync,
        TrainOut<X, OUT>: Send,
    {
        if !self.retain_gradient {
            self.gradient.set_zero();
        }
        let (send, recv) = mpsc::channel();
        let grad = batch
            .into_par_iter()
            .fold(
                || self.network.init_zero_grad(),
                |grad, p| {
                    let (input, eo) = p.as_tuple();
                    let eo = eo.borrow();
                    let (output, data) = self.train_prop(input);
                    let loss = self.loss_function.propagate(&output, eo);
                    let grad = NNTrainer::backpropagate(self, output.clone(), &eo, data, grad);

                    send.send(TrainOut { output, loss }).expect("could send output and loss");
                    grad
                },
            )
            .reduce(|| self.network.init_zero_grad(), |acc, grad| acc.add(&grad));
        self.gradient.add_mut(&grad);
        self.clip_gradient();
        self.optimize_trainee();

        recv.into_iter()
    }
}

pub struct TrainOut<X: Element, S: Shape> {
    pub output: Tensor<X, S>,
    pub loss: X,
}

impl<X, IN, OUT, L, O, NN_> fmt::Display for NNTrainer<X, IN, OUT, L, O, NN_>
where
    X: Element,
    IN: Shape,
    OUT: Shape,
    L: fmt::Display,
    O: Optimizer<X> + fmt::Debug,
    NN_: NN<X, IN, OUT>,
    [(); IN::DIM]: Sized,
    [(); OUT::DIM]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.network)?;
        write!(f, "Loss Function: {}, Optimizer: {:?}", self.loss_function, self.optimizer)
    }
}

#[cfg(test)]
mod benches;

#[cfg(test)]
mod seeded_tests {
    use crate::{
        initializer::PytorchDefault,
        loss_function::SquaredError,
        nn::{GradComponent, Pair},
        norm::Norm,
        optimizer::sgd::SGD,
        NNBuilder, NN,
    };
    use const_tensor::{tensor, Multidimensional, Vector, VectorShape};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_propagation() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let ai = NNBuilder::default()
            .double_precision()
            .rng(&mut rng)
            .input_shape::<VectorShape<2>>()
            .layer::<5>(PytorchDefault, PytorchDefault)
            .relu()
            .layer::<5>(PytorchDefault, PytorchDefault)
            .relu()
            .layer::<3>(PytorchDefault, PytorchDefault)
            .sigmoid()
            .build();

        let out = ai.prop(Vector::new(rng.gen()));

        let expected = [0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(out, tensor::literal(expected), "incorrect output");
    }

    #[test]
    fn test_gradient() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut ai = NNBuilder::default()
            .double_precision()
            .rng(&mut rng)
            .input_shape::<VectorShape<2>>()
            .layer::<5>(PytorchDefault, PytorchDefault)
            .leaky_relu(0.1)
            .layer::<5>(PytorchDefault, PytorchDefault)
            .layer::<3>(PytorchDefault, PytorchDefault)
            .sigmoid()
            .build()
            .to_trainer()
            .loss_function(SquaredError)
            .optimizer(SGD::default())
            .retain_gradient(true)
            .new_clip_gradient_norm(5.0, Norm::Two)
            .build();

        // nn params pre training
        let params = ai.get_network().iter_param().copied().collect::<Vec<_>>();
        #[rustfmt::skip]
        let expected = &[0.006887838447803829, 0.6393999304942023, 0.34936912393918684, -0.4047589840789866, -0.37941201236065963, -0.06972914538603359, 0.43380493311798884, 0.2808271748488419, -0.16417981196958276, -0.2391556648674174, 0.2108008059374471, -0.5508539013658884, 0.30609095651501483, 0.0010017747733542803, -0.2439626553503762, 0.1739790758995245, -0.35504329049611705, 0.2807057469930026, -0.021561872961492812, -0.2224985097439988, 0.18025297158732995, -0.3118176626548729, 0.26646269895835534, -0.4111905543260018, 0.07174135969857715, -0.3910179151410674, -0.14027757282776454, 0.39256214288992813, -0.1804116593475944, -0.06183204149127286, 0.30148591157620747, -0.07045111402421522, 0.15330561621693045, -0.05987140494810189, 0.16392997905786127, -0.41157175802213586, 0.06448666319062674, 0.3549482907502232, -0.1752947400236416, 0.17664346553608495, 0.4130563079306305, 0.12362639119103341, -0.4340562639542757, -0.09883618080186729, -0.05709696039076012, 0.3577843982370516, 0.1972113234741723, -0.2053210678418987, -0.03384982362548067, -0.32891932430635185, -0.26690036384241284, -0.24283061456061486, 0.23016935417459677, 0.23254520394702988, 0.3651839637543794, -0.310479259746818, -0.3017997213731933, 0.08646500039777222, -0.17584424522752867, 0.29123399909249675, 0.06853079258143152, -0.3543537884722492, 0.2413959457728258];
        assert_eq!(&params, expected, "incorrect seed");

        let pairs = (0..5)
            .map(|_| {
                let input = Vector::new(rng.gen());
                let sum = input.iter_elem().sum();
                let prod = input.iter_elem().product();
                Pair::new(input, Vector::new([sum, prod, 0.0]))
            })
            .collect::<Vec<_>>();
        ai.train_rayon(&pairs);
        //ai.train_single_thread(&pairs);

        let gradient = ai.gradient.iter_param().copied();
        #[rustfmt::skip]
        let expected = &[-0.31527687134612725, -0.2367818186318013, 0.020418606049140496, 0.015550533247454118, -0.001738594474681532, -0.004567137662788988, 0.033859035247529076, 0.02977764834407858, 0.009605083538032784, 0.006826476555722042, -0.6146889718563872, 0.040329523770678166, -0.0050435074713498255, 0.07884882135388332, 0.01821395483674612, -0.29603217316507463, 0.03346627048986953, -0.05878503376051679, -0.2158320807897315, 0.026983231976989056, -0.24554656005324604, 0.02887071882597527, -0.05843956975876532, -0.17243502863504168, 0.022467020003968326, 0.1311511999366582, -0.014868359886407069, 0.02611204347681894, 0.09796223173142266, -0.01217871097874241, -0.054034033529765414, 0.005815532263211411, -0.008911633440899904, -0.034708137931585205, 0.0043542213467064154, 0.2911033451780113, -0.033617414595422807, 0.06373660842764983, 0.21011649849556374, -0.026765687357284716, -0.6362499755696547, -0.5444227081050653, 0.288719083038049, -0.09632461875232448, 0.6412639438519863, -0.2336931253665935, -0.005400196361134401, 0.34277389227803257, -0.01683276960395113, 0.18829397721654298, 0.22312222066614518, 0.0668809841607016, -0.21284993370400693, 0.004724430001813006, -0.06979054513916816, 0.6470015756088618, 0.15347488461571596, -0.7149338885041989, 0.04221164029892477, -0.30940480020839184, -0.4461655070529444, 0.4131071925533384, 1.1799836756082676];
        println!("gradient = {:?}", ai.gradient.iter_param().collect::<Vec<_>>());
        println!("expected = {:?}", expected);
        println!(
            "diff     = {:?}",
            ai.gradient.iter_param().zip(expected).map(|(a, b)| a - b).collect::<Vec<_>>()
        );
        let err = gradient.zip(expected).map(|(p, e)| (p - e).abs()).sum::<f64>();
        println!("error = {:?}", err);
        assert!(err < 1e-15, "incorrect gradient elements (err: {})", err);
    }

    #[test]
    fn test_training() {
        const SEED: u64 = 69420;
        let mut rng = StdRng::seed_from_u64(SEED);

        let mut ai = NNBuilder::default()
            .double_precision()
            .rng(&mut rng)
            .input_shape::<VectorShape<2>>()
            .layer::<5>(PytorchDefault, PytorchDefault)
            .relu()
            .layer::<5>(PytorchDefault, PytorchDefault)
            .relu()
            .layer::<3>(PytorchDefault, PytorchDefault)
            .sigmoid()
            .build()
            .to_trainer()
            .loss_function(SquaredError)
            .optimizer(SGD::default())
            .retain_gradient(true)
            .new_clip_gradient_norm(5.0, Norm::Two)
            .build();

        // nn params pre training
        let params = ai.get_network().iter_param().copied().collect::<Vec<_>>();
        #[rustfmt::skip]
        let expected = &[0.006887838447803829, 0.6393999304942023, 0.34936912393918684, -0.4047589840789866, -0.37941201236065963, -0.06972914538603359, 0.43380493311798884, 0.2808271748488419, -0.16417981196958276, -0.2391556648674174, 0.2108008059374471, -0.5508539013658884, 0.30609095651501483, 0.0010017747733542803, -0.2439626553503762, 0.1739790758995245, -0.35504329049611705, 0.2807057469930026, -0.021561872961492812, -0.2224985097439988, 0.18025297158732995, -0.3118176626548729, 0.26646269895835534, -0.4111905543260018, 0.07174135969857715, -0.3910179151410674, -0.14027757282776454, 0.39256214288992813, -0.1804116593475944, -0.06183204149127286, 0.30148591157620747, -0.07045111402421522, 0.15330561621693045, -0.05987140494810189, 0.16392997905786127, -0.41157175802213586, 0.06448666319062674, 0.3549482907502232, -0.1752947400236416, 0.17664346553608495, 0.4130563079306305, 0.12362639119103341, -0.4340562639542757, -0.09883618080186729, -0.05709696039076012, 0.3577843982370516, 0.1972113234741723, -0.2053210678418987, -0.03384982362548067, -0.32891932430635185, -0.26690036384241284, -0.24283061456061486, 0.23016935417459677, 0.23254520394702988, 0.3651839637543794, -0.310479259746818, -0.3017997213731933, 0.08646500039777222, -0.17584424522752867, 0.29123399909249675, 0.06853079258143152, -0.3543537884722492, 0.2413959457728258];
        assert_eq!(&params, expected, "incorrect seed");

        let input = Vector::new(rng.gen());
        let eo = Vector::new(rng.gen());
        let pair = Pair::new(input, eo);

        // propagation pre training
        let prop_out = ai.propagate(pair.get_input());
        #[rustfmt::skip]
        let expected = [0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(prop_out, tensor::literal(expected), "incorrect propagation (pre training)");

        // do training
        //let res = ai.train([&pair]).execute();
        let pairs = [pair.clone()];
        let mut res_iter = ai.train_single_thread_output(&pairs);
        let res = res_iter.next().unwrap();
        assert!(res_iter.count() == 0);

        // gradient
        let gradient = ai.gradient.iter_param().copied();
        #[rustfmt::skip]
        let expected = [-0.003572457096879446, -0.0002794669624026194, 0.0, 0.0, 0.0, 0.0, 0.0004427478746224825, 3.463537847355607e-5, 0.0, 0.0, -0.0038284717751311454, 0.0, 0.0, 0.0004744767244292752, 0.0, -0.00580726500680219, 0.0, 0.0, -0.009380821731006066, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022005357571517076, 0.0, 0.0, 0.0, 0.0, 0.008226304898399563, 0.0, 0.0, 0.0, 0.0, 0.06395275520239677, 0.0, 0.0, 0.0, 0.0, -0.013618442087793855, 0.0, 0.0, 0.0, 0.0, 0.018289699457633535, 0.1421873716797208, -0.030278140178756956];

        let err = gradient.zip(expected).map(|(p, e)| (p - e).abs()).sum::<f64>();
        assert!(err < 1e-15, "incorrect gradient elements (err: {})", err);

        // training output
        let expected = [0.5571132267977859, 0.3835754220312069, 0.5254153762665995];
        assert_eq!(res.output, tensor::literal(expected), "incorrect output");

        // training loss
        assert_eq!(res.loss, 0.09546645303826229, "incorrect loss");

        // propagation post training
        let prop_out = ai.propagate(pair.get_input());
        #[rustfmt::skip]
        let expected = [0.5570843934685307, 0.38315307990347475, 0.525483857537024];
        assert_eq!(prop_out, tensor::literal(expected), "incorrect propagation (post training)");

        #[rustfmt::skip]
        const TRAINED_PARAMS: &[f64] = &[0.006923563018772624, 0.6394027251638263, 0.34936912393918684, -0.4047589840789866, -0.37941201236065963, -0.06972914538603359, 0.4338005056392426, 0.2808268284950572, -0.16417981196958276, -0.2391556648674174, 0.2108390906551984, -0.5508539013658884, 0.30609095651501483, 0.0009970300061099876, -0.2439626553503762, 0.1740371485495925, -0.35504329049611705, 0.2807057469930026, -0.021468064744182752, -0.2224985097439988, 0.18025297158732995, -0.3118176626548729, 0.26646269895835534, -0.4111905543260018, 0.07174135969857715, -0.3910179151410674, -0.14027757282776454, 0.39256214288992813, -0.1804116593475944, -0.06183204149127286, 0.30148591157620747, -0.07045111402421522, 0.15330561621693045, -0.05987140494810189, 0.16392997905786127, -0.41157175802213586, 0.06448666319062674, 0.3549482907502232, -0.1752947400236416, 0.17664346553608495, 0.41327636150634567, 0.12362639119103341, -0.4340562639542757, -0.09883618080186729, -0.05709696039076012, 0.3577021351880676, 0.1972113234741723, -0.2053210678418987, -0.03384982362548067, -0.32891932430635185, -0.2675398913944368, -0.24283061456061486, 0.23016935417459677, 0.23254520394702988, 0.3651839637543794, -0.3103430753259401, -0.3017997213731933, 0.08646500039777222, -0.17584424522752867, 0.29123399909249675, 0.06834789558685518, -0.3557756621890464, 0.24169872717461338];

        let err = ai
            .get_network()
            .iter_param()
            .copied()
            .zip(TRAINED_PARAMS)
            .map(|(p, e)| (p - e).abs())
            .sum::<f64>();
        println!("error = {:?}", err);
        assert!(err < 1e-15, "incorrect trained parameters (err: {})", err);

        let trained_params = ai.get_network().iter_param().copied().collect::<Vec<_>>();
        assert_eq!(&trained_params, TRAINED_PARAMS, "incorrect trained parameters");
    }
}
