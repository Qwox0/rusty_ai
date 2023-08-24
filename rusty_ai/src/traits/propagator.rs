use crate::{prelude::*, results::InputPropagation};

pub trait NNInput<const IN: usize> {
    fn get_input(&self) -> &[f64; IN];
}

pub trait Propagator<const IN: usize, const OUT: usize, EO> {
    fn propagate_single<'a>(&'a self, input: &'a [f64; IN]) -> PairPropagation<'a, Self, IN, OUT>;
    fn propagate_direct(&self, input: &[f64; IN]) -> [f64; OUT];

    fn propagate<'a>(&'a self, batch: &'a [[f64; IN]]) -> InputPropagation<'a, Self, IN, OUT> {
        InputPropagation::new
    }

    fn propagate_pairs<'a>(
        &'a self,
        batch: &'a [Pair<IN, EO>],
    ) -> PairPropagation<'a, Self, IN, OUT>
    where
        Self: Sized,
    {
        PairPropagation::new(self, batch)
    }

    /*
    fn test_propagate<'a>(
        &'a self,
        data_pairs: impl IntoIterator<Item = &'a Pair<IN, OUT>>,
    ) -> TestsResult<OUT>;
    */
}

fn test() {
    let mut ai = NNBuilder::default()
        .input::<{ 28 * 28 }>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::ReLU)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(10, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .log_softmax()
        .build::<10>()
        .to_trainable_builder()
        .loss_function(NLLLoss)
        .optimizer(SGD { learning_rate: 0.003, momentum: 0.9 })
        .retain_gradient(false)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    let data: PairList<10, usize> = todo!();
    let chunk_count = 13;

    for e in 0..10 {
        let mut running_loss = 0.0;
        for batch in data.chunks(64) {
            // let out = ai.propagate(batch).output();
            let out = ai.propagate(batch).backpropagate();
        }

        data.shuffle();

        println!("epoch: {}, loss: {}", e, running_loss / chunk_count as f64)
    }
}
