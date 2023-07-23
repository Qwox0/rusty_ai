#![feature(test)]

use rusty_ai::prelude::*;

pub fn main() {
    const INPUT_SIZE: usize = 28 * 28;

    let mut ai = NeuralNetworkBuilder::default()
        .default_activation_function(ActivationFn::ReLU(0.0))
        .input::<INPUT_SIZE>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(10, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::LogSoftmax)
        .error_function(ErrorFunction::SquaredError)
        .build::<10>()
        .to_trainable_builder()
        .sgd(GradientDescent { learning_rate: 0.003 })
        .retain_gradient(true)
        //.new_clip_gradient_norm(5.0, Norm::Two) // ?
        .build();

    const EPOCHS: usize = 15;

    /*
    let training_data = todo!();
    let test_data = todo!();

    let res = ai.test_propagate(test_data.iter());
    println!("epoch: {:>4}, loss: {:<20}", 0, res.error);

    ai.full_train(&training_data, EPOCHS, |epoch, ai| {
        if epoch % 100 == 0 {
            let res = ai.test_propagate(test_data.iter());
            println!("epoch: {:>4}, loss: {:<20}", epoch, res.error);
        }
    });
    */
}
