use crate::gradient::aliases::OutputGradient;

#[derive(Debug, Clone, Default)]
pub enum ErrorFunction {
    SquaredError,

    #[default]
    HalfSquaredError,

    MeanSquaredError,
}

impl ErrorFunction {
    pub fn calculate<const N: usize>(&self, output: &[f64; N], expected_output: &[f64; N]) -> f64 {
        use ErrorFunction::*;
        let squared_errors = output
            .iter()
            .zip(expected_output)
            .map(|(out, expected)| out - expected)
            .map(|err| err * err);
        match self {
            SquaredError => squared_errors.sum(),
            HalfSquaredError => 0.5 * squared_errors.sum::<f64>(),
            MeanSquaredError => 1.0 / N as f64 * squared_errors.sum::<f64>(),
        }
    }

    pub fn gradient<'a>(&self, output: Vec<f64>, expected_output: impl AsRef<[f64]>) -> OutputGradient {
        assert_eq!(output.len(), expected_output.as_ref().len());
        let errors = output
            .iter()
            .zip(expected_output.as_ref().iter())
            .map(|(out, expected)| out - expected);
        use ErrorFunction::*;
        match self {
            SquaredError => errors.map(|x| x * 2.0).collect(),
            HalfSquaredError => errors.collect(),
            MeanSquaredError => errors.map(|x| x * 2.0 / output.len() as f64).collect(),
        }
    }
}
