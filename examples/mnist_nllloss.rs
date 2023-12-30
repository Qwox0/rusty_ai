#![feature(int_roundings)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(test)]

use matrix::Num;
use mnist_util::{get_mnist, image_to_string, Mnist};
use rusty_ai::{
    data::{Pair, PairList},
    loss_function::NLLLoss,
    neural_network::NNBuilder,
    optimizer::sgd::SGD,
    ActivationFn, BuildLayer, Initializer, Norm,
};
use std::{ops::Range, time::Instant};

const IMAGE_SIDE: usize = 28;
const IMAGE_SIZE: usize = IMAGE_SIDE * IMAGE_SIDE;
const OUTPUTS: usize = 10;

const NORMALIZE_MEAN: f64 = 0.5;
const NORMALIZE_STD: f64 = 0.5;
fn transform<X: Num>(img_vec: Vec<u8>, lbl_vec: Vec<u8>) -> PairList<X, IMAGE_SIZE, usize> {
    img_vec
        .into_iter()
        .map(|x| (((x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD).cast())
        .array_chunks()
        .zip(lbl_vec.into_iter().map(usize::from))
        .collect()
}

pub fn main() {
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = get_mnist();

    let mut training_data = transform(trn_img, trn_lbl);
    let test_data = transform(tst_img, tst_lbl);

    assert_eq!(training_data.len(), 60_000);
    assert_eq!(test_data.len(), 10_000);

    println!("Example Image:");
    print_image(&training_data[50], -1.0..1.0);

    let mut ai = NNBuilder::default()
        .double_precision()
        .input::<IMAGE_SIZE>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::ReLU)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(OUTPUTS, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .log_softmax()
        .build::<OUTPUTS>()
        .to_trainer()
        .loss_function(NLLLoss)
        .optimizer(SGD { learning_rate: 0.003, momentum: 0.9 })
        .retain_gradient(false)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 15;
    const BATCH_SIZE: usize = 64;
    let batch_num = training_data.len().div_ceil(BATCH_SIZE);

    println!("\nTraining:");
    let start = Instant::now();

    for e in 0..EPOCHS {
        let mut running_loss = 0.0;
        for batch in training_data.chunks(BATCH_SIZE) {
            let loss = ai.train(batch).mean_loss();
            running_loss += loss;
        }
        // shuffle data after one full iteration
        training_data.shuffle();

        let training_loss = running_loss / batch_num as f64;

        println!("Epoch {} - Training loss: {}", e, training_loss);
        let secs = start.elapsed().as_secs();
        println!("Training Time: {} min {} s", secs / 60, secs % 60);
    }

    println!("\nTest:");
    for pair in test_data.iter().take(3) {
        print_image(pair, -1.0..1.0);
        let (input, expected_output) = pair;
        let (out, loss) = ai.test(input, expected_output);
        println!("output: {:?}", out);
        let propab = out.iter().copied().map(f64::exp).collect::<Vec<_>>();
        let guess = propab.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        println!("propab: {:?}; guess: {}", propab, guess);
        println!("error: {}", loss);
        assert!(loss < 0.2);
    }
}

pub fn print_image(pair: &Pair<f64, IMAGE_SIZE, usize>, val_range: Range<f64>) {
    println!(
        "{} label: {}",
        image_to_string::<IMAGE_SIDE, IMAGE_SIZE>(pair.0.as_ref(), val_range),
        pair.1
    );
}
