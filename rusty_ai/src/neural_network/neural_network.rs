#[allow(unused)]
use crate::neural_network::NeuralNetworkBuilder;
use crate::{
    data::Pair,
    error_function::ErrorFunction,
    gradient::Gradient,
    layer::{InputLayer, IsLayer, Layer},
    prelude::VerbosePropagation,
    results::{PropagationResult, TestsResult},
    traits::Propagator,
    util::impl_getter,
};

#[derive(Debug)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    input_layer: InputLayer<IN>,
    layers: Vec<Layer>,
    error_function: ErrorFunction,
    generation: usize,
}

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    impl_getter! { pub get_generation -> generation: usize }
    impl_getter! { pub get_layers -> layers: &Vec<Layer> }

    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(
        input_layer: InputLayer<IN>,
        layers: Vec<Layer>,
        error_function: ErrorFunction,
    ) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork {
            input_layer,
            layers,
            error_function,
            generation: 0,
        }
    }

    pub fn iter_layers(&self) -> core::slice::Iter<Layer> {
        self.layers.iter()
    }

    pub(crate) fn iter_mut_layers(&mut self) -> core::slice::IterMut<Layer> {
        self.layers.iter_mut()
    }

    pub(crate) fn increment_generation(&mut self) {
        self.generation += 1;
    }

    pub fn init_zero_gradient(&self) -> Gradient {
        self.iter_layers()
            .map(Layer::init_zero_gradient)
            .collect::<Vec<_>>()
            .into()
    }

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
        &self,
        verbose_prop: VerbosePropagation,
        expected_output: &[f64; OUT],
        gradient: &mut Gradient,
    ) {
        let mut outputs = verbose_prop.outputs;
        let last_output = outputs.pop().expect("There is an output layer");
        let expected_output = expected_output.to_vec();

        // derivatives of the cost function with respect to the output of the neurons in the last layer.
        let last_output_gradient = self.error_function.gradient(last_output, expected_output); // dC/do_L_i; i = last
        let inputs_rev = outputs.into_iter().rev();

        self.layers
            .iter()
            .zip(verbose_prop.derivatives)
            .zip(gradient.iter_mut_layers())
            .rev()
            .zip(inputs_rev)
            .fold(
                last_output_gradient,
                |current_output_gradient, (((layer, derivative_output), gradient), input)| {
                    // dc_dx = partial derivative of the cost function with respect to x.
                    let (bias_gradient, weight_gradient, input_gradient) =
                        layer.backpropagation2(derivative_output, &input, current_output_gradient);

                    /*
                    #[cfg(debug_assertions)]
                    println!("BIAS_GRAD: {:?}", bias_gradient);
                    */

                    gradient.add_next_backpropagation(weight_gradient, bias_gradient);
                    input_gradient
                },
            );
    }
}

impl<const IN: usize, const OUT: usize> Propagator<IN, OUT> for NeuralNetwork<IN, OUT> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.iter_layers()
            .fold(input.to_vec(), |acc, layer| layer.calculate(acc))
            /*
            .fold(input.to_vec(), |acc, layer| {
                let res = layer.calculate(acc);
                println!("{:?}", res);
                res
            })
            */
            .into()
    }

    fn propagate_many(&self, input_list: &Vec<[f64; IN]>) -> Vec<PropagationResult<OUT>> {
        input_list.iter().map(|x| self.propagate(x)).collect()
    }

    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT> {
        TestsResult::collect(
            data_pairs.into_iter().map(|pair| {
                let output = self.propagate(&pair.input);
                let error = self.error_function.calculate(&output.0, &pair.output);
                (output, error)
            }),
            self.generation,
        )
    }
}

impl<const IN: usize, const OUT: usize> std::fmt::Display for NeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\n{}",
            self.input_layer,
            self.layers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .join("\n")
        )?;
        if let Some(output_layer) = self.layers.last() {
            let output_count = output_layer.get_neuron_count();
            let plural_s = if output_count == 1 { "" } else { "s" };
            write!(f, "\n{} Output{}", output_count, plural_s)?;
        }
        Ok(())
    }
}

/*
#[cfg(test)]
mod tests {
    use super::NeuralNetwork;
    use crate::builder::{self, HasErrFn, NeuralNetworkBuilder, NoErrFn, NoOptimizer};
    use crate::data::DataPair;
    use crate::data_list::DataList;
    use crate::error_function::ErrorFunction;
    use crate::layer::{InputLayer, LayerType::*};
    use crate::optimizer::adam::Adam;
    use crate::optimizer::Optimizer;
    use crate::{activation_function::ActivationFunction, layer::Layer, matrix::Matrix};
    use itertools::Itertools;
    use rand::seq::SliceRandom;
    use std::iter::once;

    #[test]
    fn gradient_descent() {
        let slope = 1.5;
        let intercept = -3.0;
        const MAX_ITERATION: usize = 1000;
        const LEARNING_RATE: f64 = 0.01;

        let optimizer = Optimizer::gradient_descent(LEARNING_RATE);
        let mut ai = NeuralNetworkBuilder::new()
            .input_layer::<1>()
            .hidden_layer(3, ActivationFunction::Identity)
            //.hidden_layer(3, ActivationFunction::Identity)
            .output_layer::<1>(ActivationFunction::Identity)
            .build();

        println!("\nIteration 0: {ai}");

        let mut rng = rand::thread_rng();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = x.iter().map(|x| slope * x + intercept).collect_vec();
        let data_pairs = DataList::from_simple_vecs(x, y);

        for t in 1..=MAX_ITERATION {
            let data = data_pairs.choose_multiple(&mut rng, 5);
            ai.training_step(data, &mut optimizer);

            if t % 10usize.pow(t.ilog10()) == 0 {
                println!("\nIteration {t}: {ai}");
            }
        }

        println!("y = {} * x + {}", slope, intercept);
        let x = vec![-3.0, 0.0, 1.0, 10.0];
        println!("test for x in {:?}", x);
        let y = x.iter().map(|x| slope * x + intercept).collect();
        let test = DataList::from_simple_vecs(x, y);
        let res = ai.test_propagate(test.iter());
        println!("res: {res:?}");
        assert!(res.error.abs() < 10f64.powi(-8));
        panic!()
    }

    fn test_activation_functions(activation_function: ActivationFunction) {
        let slope = 2.0;
        let intercept = 0.0;
        const MAX_ITERATION: usize = 10000;
        const LEARNING_RATE: f64 = 0.01;

        let mut ai = NeuralNetworkBuilder::new()
            .input_layer::<1>()
            //.hidden_layer(3, activation_function)
            .hidden_layer(3, activation_function)
            .output_layer::<1>(ActivationFunction::Identity)
            .optimizer(Optimizer::gradient_descent(LEARNING_RATE))
            //.optimizer(OptimizerDispatch::Adam(Adam::default()))
            .build();

        /*
        let mut ai = NeuralNetwork::<1, 1>::new(
            InputLayer::<1>,
            vec![
            /*
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![0.1], vec![0.2], vec![0.3]], 0.0),
                    0.35,
                    activation_function,
                ),
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![0.1; 3], vec![0.2; 3], vec![0.3; 3]], 0.0),
                    0.35,
                    activation_function,
                ),
                */
            /*
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![0.1]], 0.0),
                    0.35,
                    activation_function,
                ),
                */
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![0.2]], 0.0),
                    0.35,
                    activation_function,
                ),
                Layer::new(
                    Output,
                    Matrix::from_rows(vec![vec![0.5]], 0.0),
                    0.6,
                    ActivationFunction::Identity,
                ),
            ],
            ErrorFunction::HalfSquaredError,
            OptimizerDispatch::gradient_descent(0.01),
        );
            */

        println!("\nIteration 0: {ai}");

        let mut rng = rand::thread_rng();
        let x = (-5..11).into_iter().map(f64::from).collect_vec();
        //let x = vec![5.0, 3.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        //let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        //let x = vec![-3.0, 0.0, 1.0, 10.0];
        let y = x.iter().map(|x| slope * x + intercept).collect_vec();
        let data_pairs = DataList::from_simple_vecs(x, y);

        for t in 1..=MAX_ITERATION {
            //let data = &data_pairs[t];
            //ai.train_single(once(data));
            let data = data_pairs.choose_multiple(&mut rng, 5);
            // Changing the 5 to a 1 results in inaccurate gradients which cause exploding weight values!
            ai.train_single(data);

            if t % 10usize.pow(t.ilog10()) == 0 {
                println!("\nIteration {t}: {ai}");
            }
        }

        println!("y = {} * x + {}", slope, intercept);
        let x = vec![-3.0, 0.0, 1.0, 10.0];
        println!("test for x in {:?}", x);
        let y = x.iter().map(|x| slope * x + intercept).collect();
        let test = DataList::from_simple_vecs(x, y);
        let res = ai.test_propagate(test.iter());
        println!("res: {res:?}");
        assert!(res.error.abs() < 10f64.powi(-8));
    }

    #[test]
    fn act_fn_identity() {
        test_activation_functions(ActivationFunction::Identity)
    }

    #[test]
    fn act_fn_sigmoid() {
        test_activation_functions(ActivationFunction::Sigmoid)
    }

    #[test]
    fn act_fn_relu() {
        test_activation_functions(ActivationFunction::default_relu())
    }

    #[test]
    fn act_fn_leaky_relu() {
        test_activation_functions(ActivationFunction::default_leaky_relu())
    }

    /// test backpropagation example calculation from
    /// `https://medium.com/edureka/backpropagation-bd2cf8fdde81`
    #[test]
    fn backpropagation1() {
        let optimizer = Optimizer::gradient_descent(0.5);
        let mut ai = NeuralNetwork::<2, 2>::new(
            InputLayer::<2>,
            vec![
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![0.15, 0.2], vec![0.25, 0.3]], 0.0),
                    0.35,
                    ActivationFunction::Sigmoid,
                ),
                Layer::new(
                    Output,
                    Matrix::from_rows(vec![vec![0.4, 0.45], vec![0.5, 0.55]], 0.0),
                    0.6,
                    ActivationFunction::Sigmoid,
                ),
            ],
            ErrorFunction::HalfSquaredError,
        );

        println!("{:.4}\n", ai);

        let data_pair = DataPair::from(([0.05, 0.1], [0.01, 0.99]));
        println!("{data_pair}");

        let res = ai.test_propagate(once(&data_pair));
        println!("{:?}\n\n", res);
        assert!((res.outputs[0][0] - 0.75136507).abs() < 10f64.powi(-8));
        assert!((res.outputs[0][1] - 0.772928465).abs() < 10f64.powi(-8));
        // 0.5: different formula for
        assert!((res.error - 0.298371109).abs() < 10f64.powi(-8));

        ai.training_step(once(&data_pair), &mut optimizer);
        println!("{:.8}\n", ai);

        let weight5 = ai.layers.last().unwrap().get_weights().get(0, 0).unwrap();
        let err = (weight5 - 0.35891648).abs();

        println!("err: {}", err);
        assert!(err < 10f64.powi(-8));
    }

    /// test backpropagation example calculation from
    /// `https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html`
    #[test]
    fn backpropagation2() {
        const LEARNING_RATE: f64 = 0.1;
        let mut ai = NeuralNetwork::<2, 2>::new(
            InputLayer::<2>,
            vec![
                Layer::new(
                    Hidden,
                    Matrix::from_rows(vec![vec![6.0, -2.0], vec![-3.0, 5.0]], 0.0),
                    0.0,
                    ActivationFunction::Sigmoid,
                ),
                Layer::new(
                    Output,
                    Matrix::from_rows(vec![vec![1.0, 0.25], vec![-2.0, 2.0]], 0.0),
                    0.0,
                    ActivationFunction::Sigmoid,
                ),
            ],
            ErrorFunction::SquaredError,
        );
        let optimizer = Optimizer::gradient_descent(LEARNING_RATE);

        println!("{:.4}\n", ai);

        let data_pair1 = DataPair::from(([3.0, 1.0], [1.0, 0.0]));
        println!("{}", data_pair1);
        let res = ai.test_propagate(once(&data_pair1));
        assert!((res.outputs[0][0] - 0.73).abs() < 10f64.powi(-2));
        assert!((res.outputs[0][1] - 0.12).abs() < 10f64.powi(-2));
        assert!((res.error - 0.08699208259994781).abs() < 10f64.powi(-8));

        ai.training_step(once(&data_pair1), &mut optimizer);
        println!("{:.8}\n", ai);

        let hidden_layer1 = ai.layers[0].get_weights();

        let w12 = hidden_layer1[(1, 0)];
        println!("{w12}");
        let err = (w12 - (-3.0 - LEARNING_RATE * 0.0014201436720081408)).abs();
        println!("err: {}", err);
        assert!(err < 10f64.powi(-8));

        let w22 = hidden_layer1[(1, 1)];
        println!("{w22}");
        let err = (w22 - (5.0 - LEARNING_RATE * 0.0004733812240027136)).abs();
        println!("err: {}", err);
        assert!(err < 10f64.powi(-8));
    }

    /// https://www.youtube.com/watch?v=6nqV58NA_Ew
    fn adam_fit(slope: f64, intercept: f64) {
        const MAX_ITERATION: usize = 3000;
        const LEARNING_RATE: f64 = 0.01;
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPSILON: f64 = 0.00000001;

        let mut ai = NeuralNetworkBuilder::new()
            .input_layer::<1>()
            //.hidden_layer(1, ActivationFunction::Identity)
            .hidden_layer(3, ActivationFunction::Identity)
            .output_layer::<1>(ActivationFunction::Identity)
            //.optimizer(OptimizerDispatch::gradient_descent(0.01))
            .optimizer(Optimizer::Adam(Adam::new(
                LEARNING_RATE,
                BETA1,
                BETA2,
                EPSILON,
            )))
            .build();

        println!("\nIteration 0: {ai}");

        let mut rng = rand::thread_rng();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = x.iter().map(|x| slope * x + intercept).collect_vec();
        let data_pairs = DataList::from_simple_vecs(x, y);

        for t in 1..=MAX_ITERATION {
            let data = data_pairs.choose_multiple(&mut rng, 5);
            ai.train_single(data);

            if t % 10usize.pow(t.ilog10()) == 0 {
                println!("\nIteration {t}: {ai}");
            }
        }

        println!("y = {} * x + {}", slope, intercept);
        let x = vec![-3.0, 0.0, 1.0, 10.0];
        println!("test for x in {:?}", x);
        let y = x.iter().map(|x| slope * x + intercept).collect();
        let test = DataList::from_simple_vecs(x, y);
        let res = ai.test_propagate(test.iter());
        println!("res: {res:?}");
        assert!(res.error.abs() < 10f64.powi(-8));
    }

    #[test]
    fn adam_fit_slope1_intercept0() {
        adam_fit(1.0, 0.0)
    }

    #[test]
    fn adam_fit_slope2_intercept0() {
        adam_fit(2.0, 0.0)
    }

    #[test]
    fn adam_fit_slope2_intercept1() {
        adam_fit(2.0, 1.0)
    }
}
*/