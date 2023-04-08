use crate::{
    layer::Layer,
    results::{PropagationResult, TestsResult},
};

#[derive(Debug)]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
    generation: usize,
}

#[allow(unused)]
use crate::builder::NeuralNetworkBuilder;

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// use [`NeuralNetworkBuilder`] instead!
    pub(crate) fn new(layers: Vec<Layer>) -> NeuralNetwork<IN, OUT> {
        NeuralNetwork {
            layers,
            generation: 0,
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

    pub fn train(&mut self, data_pairs: impl Iterator<Item = ([f64; IN], [f64; OUT])>) {
        todo!()
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
