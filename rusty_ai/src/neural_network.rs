use crate::{
    data::DataPair,
    data_list::DataList,
    error_function::ErrorFunction,
    layer::{InputLayer, Layer},
    optimizer::{Optimizer, OptimizerDispatch},
    results::{GradientLayer, PropagationResult, TestsResult},
    util::{EntryDiv, EntrySub, ScalarMul},
};

#[derive(Debug)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    input_layer: InputLayer<IN>,
    layers: Vec<Layer>,
    error_function: ErrorFunction,
    optimizer: OptimizerDispatch,
    generation: usize,
}

pub struct NNOptimizationParts<'a> {
    pub layers: &'a mut Vec<Layer>,
    pub generation: &'a usize,
}

#[allow(unused)]
use crate::builder::NeuralNetworkBuilder;

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(
        input_layer: InputLayer<IN>,
        layers: Vec<Layer>,
        error_function: ErrorFunction,
        optimizer: OptimizerDispatch,
    ) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork {
            input_layer,
            layers,
            error_function,
            optimizer,
            generation: 0,
        }
    }

    pub fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.layers
            .iter()
            .fold(input.to_vec(), |acc, layer| layer.calculate(acc))
            .into()
    }

    pub fn propagate_many(&self, input_list: &Vec<[f64; IN]>) -> Vec<PropagationResult<OUT>> {
        input_list
            .iter()
            .map(|input| self.propagate(input))
            .collect()
    }

    pub fn test2<'a>(
        &'a self,
        inputs: &'a Vec<[f64; IN]>,
        expected_outputs: &'a Vec<[f64; OUT]>,
    ) -> TestsResult<IN, OUT> {
        assert_eq!(inputs.len(), expected_outputs.len());
        let outputs = self.propagate_many(inputs);
        let error = outputs
            .iter()
            .zip(expected_outputs)
            .map(|(out, expected)| self.error_function.calculate(&out.0, expected))
            .sum();
        TestsResult {
            generation: self.generation,
            outputs: outputs.into_iter().map(Into::into).collect(),
            error,
        }
    }
    pub fn test<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a DataPair<IN, OUT>>,
    ) -> TestsResult<IN, OUT> {
        let (outputs, error) = data_pairs
            .into_iter()
            .map(|pair| {
                let output = self.propagate(&pair.input);
                let err = self.error_function.calculate(&output.0, &pair.output);
                (output, err)
            })
            .fold((vec![], 0.0), |mut acc, (output, err)| {
                acc.0.push(output.into());
                acc.1 += err;
                acc
            });
        TestsResult {
            generation: self.generation,
            outputs,
            error,
        }
    }

    fn training_propagate(&self, input: &[f64; IN]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let res = self.layers.iter().fold(
            (vec![input.to_vec()], Vec::with_capacity(self.layers.len())),
            |(mut outputs, mut derivatives), layer| {
                let input = outputs.last().expect("last element must exists");
                let (output, derivative) = layer.training_calculate(input);
                outputs.push(output);
                derivatives.push(derivative);
                (outputs, derivatives)
            },
        );
        assert!(res.0.len() == self.layers.len() + 1);
        assert!(res.1.len() == self.layers.len());
        res
    }
    fn training_propagate2(&self, input: &[f64; IN]) -> (Vec<Vec<f64>>, Vec<Option<Vec<f64>>>) {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut derivatives = Vec::with_capacity(self.layers.len());
        outputs.push(input.to_vec());
        derivatives.push(None);

        for i in 0..self.layers.len() {
            let input = outputs.last().expect("last element must exists");
            let (output, derivative) = self.layers[i].training_calculate(input);
            outputs.push(output);
            derivatives.push(Some(derivative));
        }
        assert!(outputs.len() == self.layers.len() + 1);
        assert!(derivatives.len() == self.layers.len() + 1);
        (outputs, derivatives)
    }

    pub fn train(
        &mut self,
        training_data: &DataList<IN, OUT>,
        training_amount: usize,
        epoch_count: usize,
        silent: bool,
    ) {
        let mut rng = rand::thread_rng();
        for epoch in 0..epoch_count {
            let training_data = training_data.choose_multiple(&mut rng, training_amount);
            self.train_single(training_data);
            if !silent {
                println!("{}", epoch);
            }
        }
    }

    /// Trains the neural network for one generation/epoch. Uses a small data set `data_pairs` to
    /// find an aproximation for the weights gradient. The neural network's Optimizer changes the
    /// weights by using the calculated gradient.
    pub fn train_single<'a>(
        &mut self,
        data_pairs: impl IntoIterator<Item = &'a DataPair<IN, OUT>>,
    ) {
        let mut data_count = 0;
        // estimated gradient, but seperated for each (non input) layer. starts with last layer,
        // ends with second to first
        let mut gradient = self.layers.iter().rev().map(Layer::init_gradient).collect();
        for (input, expected_output) in data_pairs.into_iter().map(Into::into) {
            let (all_outputs, all_derivatives) = self.training_propagate(&input);

            #[cfg(debug_assertions)]
            {
                let get_padding =
                    |l| " ".repeat(80usize.checked_sub(format!("{l:?}").len()).unwrap_or(2));
                let mut out_iter = all_outputs.iter();
                let input = out_iter.next().unwrap();
                println!(
                    "PROPAGATE:\n   {:?}\n   {}",
                    input,
                    out_iter
                        .zip(all_derivatives.iter())
                        .map(|(l, dl)| format!("{:?} {}; {:?}", l, get_padding(l), dl))
                        .collect::<Vec<_>>()
                        .join("\n   ")
                );
            }

            self.backpropagation(all_outputs, expected_output, all_derivatives, &mut gradient);
            data_count += 1;
        }
        gradient.mut_mul_scalar(1.0 / data_count as f64);

        #[cfg(debug_assertions)]
        println!("GRADIENT: {:.4?}", gradient);

        self.optimize_weights(gradient);
        self.generation += 1;
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
    fn backpropagation(
        &self,
        outputs: Vec<Vec<f64>>,
        expected_output: &[f64; OUT], // only for the last layer
        derivative_outputs: Vec<Vec<f64>>,
        gradient: &mut Vec<GradientLayer>,
    ) {
        let mut outputs = outputs.into_iter().rev();

        // derivatives of the cost function with respect to the output of the neurons in the last layer.
        let last_output_gradient = self
            .error_function
            .gradient(outputs.next().unwrap(), expected_output.to_vec());

        let inputs_rev = outputs;

        self.layers
            .iter()
            .zip(derivative_outputs)
            .rev()
            .zip(inputs_rev)
            .zip(gradient.iter_mut())
            .fold(
                last_output_gradient,
                |current_output_gradient, (((layer, derivative_output), input), gradient)| {
                    // dc_dx = partial derivative of the cost function with respect to x.
                    let (bias_change, weight_gradient, inputs_gradient) =
                        layer.backpropagation2(derivative_output, &input, current_output_gradient);

                    gradient.add_next_backpropagation(weight_gradient, bias_change);
                    inputs_gradient
                },
            );
    }

    fn optimize_weights(&mut self, gradient: Vec<GradientLayer>) {
        self.optimizer.optimize_weights(
            NNOptimizationParts {
                layers: &mut self.layers,
                generation: &self.generation,
            },
            gradient,
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
        )
    }
}

#[cfg(test)]
mod tests {
    use super::NeuralNetwork;
    use crate::builder::NeuralNetworkBuilder;
    use crate::data::DataPair;
    use crate::data_list::DataList;
    use crate::error_function::ErrorFunction;
    use crate::layer::{InputLayer, LayerType::*};
    use crate::optimizer::adam::Adam;
    use crate::optimizer::OptimizerDispatch;
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

        let mut ai = NeuralNetworkBuilder::new()
            .input_layer::<1>()
            .hidden_layer(3, ActivationFunction::Identity)
            //.hidden_layer(3, ActivationFunction::Identity)
            .output_layer::<1>(ActivationFunction::Identity)
            .optimizer(OptimizerDispatch::gradient_descent(LEARNING_RATE))
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
        let res = ai.test(test.iter());
        println!("res: {res:?}");
        assert!(res.error.abs() < 10f64.powi(-8));
    }

    #[test]
    fn neural_network() {
        /*
        let mut ai = NeuralNetwork::<1, 1>::new(vec![
            Layer::new_input(1),
            Layer::new(
                Hidden,
                Matrix::from_rows(vec![vec![0.01], vec![1.0], vec![0.01]], 0.0),
                0.0,
                ActivationFunction::default_relu(),
            ),
            Layer::new(
                Output,
                Matrix::from_rows(vec![vec![0.0, 2.0, 1.0]], 0.0),
                //Matrix::from_rows(vec![vec![-2.0]], 0.0),
                00.0,
                ActivationFunction::Identity,
            ),
        ]);
        */
        let mut ai = NeuralNetworkBuilder::new()
            .input_layer::<1>()
            .hidden_layer(5, ActivationFunction::default_relu())
            //.hidden_layer(3, ActivationFunction::default_relu())
            .output_layer::<1>(ActivationFunction::Identity)
            .optimizer(OptimizerDispatch::gradient_descent(0.5))
            .build();

        // one hidden layer: Gen10: TestsResult { generation: 91, outputs: [[-8.670479574319565], [3.428770522035781], [3.152536785898278], [3.428770522035781]], error: 7.036894229866931 }
        //
        println!("{}\n", ai);

        //let mut res: Vec<_> = vec![];

        const DATA_COUNT: usize = 1;
        let get_data = || {
            DataList::from(
                rand::random::<[f64; DATA_COUNT]>()
                    .into_iter()
                    .map(|x| x - 0.5)
                    .map(|x| x * 20.0)
                    .map(|x| (x, x * 2.0))
                    .collect::<Vec<_>>(),
            )
        };
        /*

        let format = |vec_x: Vec<f64>| {
            vec_x
                .into_iter()
                .map(|x| ([x], [x * 2.0]))
                .collect::<Vec<_>>()
        };


        res.push(ai.test2(
            &vec![[-5.0], [1.0], [0.0], [2.0]],  // [100000.0]],
            &vec![[-10.0], [2.0], [0.0], [4.0]], // [200000.0]],
        ));

        ai.train(format(vec![1.0, 2.0]));

        res.push(ai.test2(
            &vec![[-5.0], [1.0], [0.0], [2.0]],  // [100000.0]],
            &vec![[-10.0], [2.0], [0.0], [4.0]], // [200000.0]],
        ));

        ai.train(format(vec![0.5, 4.0]));

        res.push(ai.test2(
            &vec![[-5.0], [1.0], [0.0], [2.0]],  // [100000.0]],
            &vec![[-10.0], [2.0], [0.0], [4.0]], // [200000.0]],
        ));

        ai.train(format(vec![2.0, 1.0]));

        res.push(ai.test2(
            &vec![[-5.0], [1.0], [0.0], [2.0]],  // [100000.0]],
            &vec![[-10.0], [2.0], [0.0], [4.0]], // [200000.0]],
        ));
        */

        for i in 0..100 {
            let data = get_data();
            println!("\n\n{}\nai: {ai}\nDATA: {}", i + 1, data[0]);
            ai.train_single(data.iter());
            println!(
                "TEST: {:?}\n",
                ai.test2(
                    &vec![[-5.0], [1.0], [0.000001], [2.0]],  // [100000.0]],
                    &vec![[-10.0], [2.0], [0.000002], [4.0]], // [200000.0]],
                )
            );
            /*
            if (i + 1) % 100 == 0 {
                let a = ai.test2(
                    &vec![[-5.0], [1.0], [0.000001], [2.0]],  // [100000.0]],
                    &vec![[-10.0], [2.0], [0.000002], [4.0]], // [200000.0]],
                );
                //println!("ai:{:?}", ai);
                //println!("-> res: {:?}", a);
                res.push(a);
            }
            println!("{ai:?}");
            println!("{:?}", ai.training_propagate(&data.get(0).unwrap().0))
            */
        }
        /*

        for res in res {
            println!("{:?}", res);
        }
        println!("inputs: {:?}", vec![[-5.0], [1.0], [0.0], [2.0]]);
        println!("expected_outputs: {:?}", vec![[-10.0], [2.0], [0.0], [4.0]]);

        println!("{}", ai);

        println!("{:?}", get_data());
         */

        println!("END!!!!!!");
        panic!("END!!!!!!");
    }

    /// test backpropagation example calculation from
    /// `https://medium.com/edureka/backpropagation-bd2cf8fdde81`
    #[test]
    fn backpropagation1() {
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
            OptimizerDispatch::gradient_descent(0.5),
        );

        println!("{:.4}\n", ai);

        let data_pair = DataPair::from(([0.05, 0.1], [0.01, 0.99]));
        println!("{data_pair}");

        let res = ai.test(once(&data_pair));
        println!("{:?}\n\n", res);
        assert!((res.outputs[0][0] - 0.75136507).abs() < 10f64.powi(-8));
        assert!((res.outputs[0][1] - 0.772928465).abs() < 10f64.powi(-8));
        // 0.5: different formula for
        assert!((res.error - 0.298371109).abs() < 10f64.powi(-8));

        ai.train_single(once(&data_pair));
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
            OptimizerDispatch::gradient_descent(LEARNING_RATE),
        );

        println!("{:.4}\n", ai);

        let data_pair1 = DataPair::from(([3.0, 1.0], [1.0, 0.0]));
        println!("{}", data_pair1);
        let res = ai.test(once(&data_pair1));
        assert!((res.outputs[0][0] - 0.73).abs() < 10f64.powi(-2));
        assert!((res.outputs[0][1] - 0.12).abs() < 10f64.powi(-2));
        assert!((res.error - 0.08699208259994781).abs() < 10f64.powi(-8));

        ai.train_single(once(&data_pair1));
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
            .optimizer(OptimizerDispatch::Adam(Adam::new(
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
        let res = ai.test(test.iter());
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
