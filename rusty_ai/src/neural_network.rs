use crate::{
    layer::Layer,
    results::{GradientLayer, PropagationResult, TestsResult},
};

#[derive(Debug)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
    generation: usize,
    #[allow(unused)]
    optimizer: (), // TODO
}

#[allow(unused)]
use crate::builder::NeuralNetworkBuilder;

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(layers: Vec<Layer>) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork {
            layers,
            generation: 0,
            optimizer: (),
        }
    }

    pub fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.layers
            .iter()
            .skip(1) // first layer is always an input layer which doesn't mutate the input
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
            .map(|(out, expected)| out.mean_squarred_error(expected))
            .sum();
        TestsResult {
            generation: self.generation,
            outputs: outputs.into_iter().map(Into::into).collect(),
            error,
        }
    }
    pub fn test<'a>(
        &'a self,
        data_pairs: &'a Vec<([f64; IN], [f64; OUT])>,
    ) -> TestsResult<IN, OUT> {
        let (outputs, error) = data_pairs
            .iter()
            .map(|(input, expected_output)| {
                let output = self.propagate(input);
                let err = output.mean_squarred_error(expected_output);
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

    fn training_propagate(&self, input: &[f64; IN]) -> (Vec<Vec<f64>>, Vec<Option<Vec<f64>>>) {
        let mut vec = Vec::with_capacity(self.layers.len() - 1);
        vec.push(None); // no derivatives for the Input Layer
        let res = self
            .layers
            .iter()
            .skip(1) // first layer is always an input layer which doesn't mutate the input
            .fold(
                (vec![input.to_vec()], vec),
                |(mut outputs, mut derivatives), layer| {
                    let input = outputs.last().expect("last element must exists");
                    let (output, derivative) = layer.training_calculate(input);
                    outputs.push(output);
                    derivatives.push(Some(derivative));
                    (outputs, derivatives)
                },
            );
        assert!(res.0.len() == self.layers.len());
        assert!(res.1.len() == self.layers.len());
        res
    }

    /// uses a small data set to find an aproximation for the weights gradient.
    pub fn train(&mut self, data_pairs: impl IntoIterator<Item = ([f64; IN], [f64; OUT])>) {
        // estimated gradient, but seperated for each (non input) layer. starts with last layer,
        // ends with second to first
        let mut gradient = self
            .layers
            .iter()
            .skip(1)
            .rev()
            .map(Layer::init_gradient)
            .collect();
        for (input, expected_output) in data_pairs {
            let (all_outputs, all_derivatives) = self.training_propagate(&input);

            println!(
                "PROPAGATE:\n   {}",
                all_outputs
                    .iter()
                    .zip(all_derivatives.iter())
                    .map(|(l, dl)| format!(
                        "{:?} {}; {:?}",
                        l,
                        " ".repeat(100usize.checked_sub(format!("{l:?}").len()).unwrap_or(2)),
                        dl
                    ))
                    .collect::<Vec<_>>()
                    .join("\n   ")
            );

            self.backpropagation(all_outputs, expected_output, all_derivatives, &mut gradient);
        }
        println!("GRADIENT: {:.4?}", gradient);
        assert!(self.layers.len() - 1 == gradient.len());
        crate::optimizer::optimize_weights(self.layers.iter_mut().skip(1).rev(), gradient);
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
        expected_output: [f64; OUT], // only for the last layer
        derivative_outputs: Vec<Option<Vec<f64>>>,
        gradient: &mut Vec<GradientLayer>,
    ) {
        let mut expected_output = expected_output.to_vec();
        self.layers
            .iter()
            .zip(derivative_outputs)
            .skip(1) // Input Layer isn't affected by backpropagation
            .zip(outputs.windows(2))
            .rev()
            .enumerate()
            .map(|(idx, ((l, d), win))| (idx, l, d, win.get(0).unwrap(), win.get(1).unwrap()))
            .for_each(
                |(rev_layer_idx, layer, derivative_output, layer_input, layer_output)| {
                    let derivative_output = derivative_output.expect("layer has derivatives");

                    // dc_dx = partial derivative of the cost function with respect to x.
                    let (sum_dc_dbias, dc_dweights, sum_dc_dinputs) = layer.backpropagation2(
                        layer_input,
                        layer_output,
                        derivative_output,
                        &expected_output,
                    );

                    expected_output = sum_dc_dinputs;
                    gradient
                        .get_mut(rev_layer_idx)
                        .expect("gradient layer exists")
                        .add_next_backpropagation(dc_dweights, sum_dc_dbias);
                },
            );
    }
}

impl<const IN: usize, const OUT: usize> std::fmt::Display for NeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
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
    use crate::layer::LayerType::*;
    use crate::{activation_function::ActivationFunction, layer::Layer, matrix::Matrix};

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
            .build();

        // one hidden layer: Gen10: TestsResult { generation: 91, outputs: [[-8.670479574319565], [3.428770522035781], [3.152536785898278], [3.428770522035781]], error: 7.036894229866931 }
        //
        println!("{}\n", ai);

        //let mut res: Vec<_> = vec![];

        let get_data = || {
            rand::random::<[f64; 1]>()
                .into_iter()
                .map(|x| x - 0.5)
                .map(|x| x * 20.0)
                .map(|x| ([x], [x * 2.0]))
                .collect::<Vec<_>>()
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
            println!(
                "\n\n{}\nai: {ai}\nDATA: {} -> {}",
                i + 1,
                data[0].0[0],
                data[0].1[0]
            );
            ai.train(data.clone());
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
    fn neural_network2() {
        let mut ai = NeuralNetwork::<2, 2>::new(vec![
            Layer::new_input(2),
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
        ]);

        println!("{:.4}\n", ai);

        let data_pairs = vec![([0.05, 0.1], [0.01, 0.99])];
        println!("[0.05, 0.1] -> [0.01, 0.99]");

        let res = ai.test(&data_pairs);
        println!("{:?}\n\n", res);

        ai.train(data_pairs.clone().into_iter());
        println!("{:.8}\n", ai);

        let weight5 = ai.layers.last().unwrap().get_weights().get(0, 0).unwrap();
        let expected = 0.35891648;
        let err = (weight5 - expected).abs();

        println!("err: {}", err);
        assert!(err < 10f64.powi(-8));
    }
}
