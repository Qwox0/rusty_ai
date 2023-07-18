use rusty_ai::prelude::*;

fn main() {
    fn vec<const W: usize, const H: usize>(arr: [[f64; W]; H]) -> Vec<Vec<f64>> {
        arr.into_iter().map(Vec::from).collect()
    }

    let mut ai = NeuralNetworkBuilder::default()
        .default_activation_function(ActivationFn::ReLU(0.0))
        .default_initializer(PytorchDefault)
        .input()
        .random_layer(1)
        .random_layer(5)
        .random_layer(5)
        .random_layer(5)
        .default_activation_function(ActivationFn::Identity)
        .random_layer(1)
        .error_function(ErrorFunction::SquaredError)
        .build()
        .to_trainable_builder()
        .sgd(GradientDescent { learning_rate: 0.01 })
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    let data = DataBuilder::default().build::<1>(100).gen_pairs(|input| [input[0].sin()]);

    for epoch in 1..=10000 {
        ai.training_step(data.iter());
        if epoch % 1000 == 0 {
            let test = ai.test_propagate(data.iter());
            println!("epoch: {}, loss: {}", epoch, test.error);
        }
    }
}
