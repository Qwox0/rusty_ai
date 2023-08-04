use crate::prelude::*;

#[derive(Debug, Clone, Default)]
pub enum ErrorFunction {
    SquaredError,

    #[default]
    HalfSquaredError,

    MeanSquaredError,

    /// TODO: impove this
    NLLLoss,
}

fn squared_errors<'a, const N: usize>(
    output: &'a [f64; N],
    expected_output: &'a [f64; N],
) -> impl Iterator<Item = f64> + 'a {
    output.iter().zip(expected_output).map(|(out, expected)| out - expected).map(|err| err * err)
}

impl ErrorFunction {
    pub fn calculate<const N: usize>(&self, output: &[f64; N], expected_output: &[f64; N]) -> f64 {
        use ErrorFunction::*;
        match self {
            SquaredError => squared_errors(output, expected_output).sum(),
            HalfSquaredError => 0.5 * squared_errors(output, expected_output).sum::<f64>(),
            MeanSquaredError => {
                1.0 / N as f64 * squared_errors(output, expected_output).sum::<f64>()
            },
            NLLLoss => nllloss(output, expected_output[0].round() as usize) // TODO: improve
        }
    }

    pub fn gradient<'a>(
        &self,
        output: &Vec<f64>,
        expected_output: impl AsRef<[f64]>,
    ) -> OutputGradient {
        assert_eq!(output.len(), expected_output.as_ref().len());
        let errors = output
            .iter()
            .zip(expected_output.as_ref().iter())
            .map(|(out, expected)| out - expected);
        //.map(|(out, expected)| expected - out);
        use ErrorFunction::*;
        match self {
            SquaredError => errors.map(|x| x * 2.0).collect(),
            HalfSquaredError => errors.collect(),
            MeanSquaredError => errors.map(|x| x * 2.0 / output.len() as f64).collect(),
            NLLLoss => todo!(),
        }
    }
}

/// see [`https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html`]
pub fn nllloss<const N: usize>(output: &[f64; N], expected_output: usize) -> f64 {
    assert!((0..N).contains(&expected_output));
    -output[expected_output]
}
