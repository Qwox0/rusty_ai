mod builder;
mod trainable;
pub use builder::*;
pub use trainable::TrainableNeuralNetwork;

use crate::{
    data::Pair,
    error_function::ErrorFunction,
    gradient::Gradient,
    layer::Layer,
    results::{PropagationResult, TestsResult},
    traits::{IterLayerParams, Propagator},
    util::impl_getter,
};

#[derive(Debug)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
    error_function: ErrorFunction,
    generation: usize,
}

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    impl_getter! { pub get_generation -> generation: usize }
    impl_getter! { pub get_layers -> layers: &Vec<Layer> }

    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(layers: Vec<Layer>, error_function: ErrorFunction) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork {
            layers,
            error_function,
            generation: 0,
        }
    }

    pub fn iter_layers(&self) -> core::slice::Iter<Layer> {
        self.layers.iter()
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
}

impl<const IN: usize, const OUT: usize> IterLayerParams for NeuralNetwork<IN, OUT> {
    type Layer = Layer;

    fn iter_layers<'a>(&'a self) -> impl Iterator<Item = &'a Self::Layer> {
        self.layers.iter()
    }

    fn iter_mut_layers<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Layer> {
        self.layers.iter_mut()
    }
}

impl<const IN: usize, const OUT: usize> Propagator<IN, OUT> for NeuralNetwork<IN, OUT> {
    fn propagate(&self, input: &[f64; IN]) -> PropagationResult<OUT> {
        self.iter_layers()
            .fold(input.to_vec(), |acc, layer| layer.calculate(acc))
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
        let get_plural_s = |x: usize| if x == 1 { "" } else { "s" };
        writeln!(
            f,
            "Neural Network: {IN} Input{} -> {OUT} Output{}",
            get_plural_s(IN),
            get_plural_s(OUT),
        )?;
        let layers_text = self
            .layers
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "{}", layers_text)
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
