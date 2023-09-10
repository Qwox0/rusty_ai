#![feature(int_roundings)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(test)]

use mnist_nllloss_example::{get_data, image_to_string};
use rusty_ai::prelude::*;
use std::{iter::once, ops::Range, time::Instant};

const IMAGE_SIDE: usize = 28;
const IMAGE_SIZE: usize = IMAGE_SIDE * IMAGE_SIDE;
const OUTPUTS: usize = 10;

const NORMALIZE_MEAN: f64 = 0.5;
const NORMALIZE_STD: f64 = 0.5;
fn transform(img_vec: Vec<u8>, lbl_vec: Vec<u8>) -> PairList<IMAGE_SIZE, [f64; OUTPUTS]> {
    img_vec
        .into_iter()
        .map(|x| ((x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD)
        .array_chunks()
        .zip(lbl_vec.into_iter().map(|x| {
            let mut out = [0.0; OUTPUTS];
            out[x as usize] = 1.0;
            out
        }))
        .collect()
}

fn setup_ai() -> NNTrainer<IMAGE_SIZE, OUTPUTS, SquaredError, SGD_> {
    NNBuilder::default()
        .input::<IMAGE_SIZE>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::ReLU)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(OUTPUTS, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .sigmoid()
        .build::<OUTPUTS>()
        .to_trainer()
        .loss_function(SquaredError)
        .optimizer(SGD { learning_rate: 0.003, momentum: 0.9 })
        .retain_gradient(false)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build()
}

pub fn main() {
    let load_mnist::Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = get_data();

    let mut training_data = transform(trn_img, trn_lbl);
    let test_data = transform(tst_img, tst_lbl);

    assert_eq!(training_data.len(), 60_000);
    assert_eq!(test_data.len(), 10_000);

    println!("Example Image:");
    print_image(&training_data[50], -1.0..1.0);

    let mut ai = setup_ai();

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
        let guess = propab
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0;
        println!("propab: {:?}; guess: {}", propab, guess);
        println!("error: {}", loss);
        assert!(loss < 0.2);
    }
}

pub fn print_image(pair: &Pair<IMAGE_SIZE, [f64; 10]>, val_range: Range<f64>) {
    println!("{} label: {:?}", image_to_string(&pair.0, val_range), pair.1);
}

pub mod tests {
    extern crate test;
    use super::*;
    use test::*;

    fn load_data() -> Pair<IMAGE_SIZE, u8> {
        (
            TEST_IMG
                .iter()
                .map(|x| ((*x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD)
                .array_chunks()
                .next()
                .unwrap(),
            TEST_LBL,
        )
    }

    fn load_data_hack() -> Pair<IMAGE_SIZE, [f64; 10]> {
        let (input, output) = load_data();
        let idx = output as usize;
        let mut output = [0.0; 10];
        output[idx] = 1.0;
        (input, output)
    }

    /// #[bench]
    pub fn test_propagate(b: &mut Bencher) {
        let ai = setup_ai();
        let p = load_data();

        b.iter(|| {
            black_box(ai.propagate(black_box(&p.0)));
        })
    }

    /*
    /// #[bench]
    pub fn test_train_hack(b: &mut Bencher) {
        let mut ai = setup_ai();
        let p = load_data_hack();

        b.iter(|| {
            black_box(ai.training_step(black_box(once(&p))));
        })
    }
    */

    /// ...
    /// ...,   ,    ,    ,  12,  56, 140, 126, 175, 200,  96,   2,    ,    ,    , ...
    /// ...,   ,  35, 166, 238, 254, 246, 242, 253, 246, 254,  67,    ,    ,    , ...
    /// ...,   , 184, 182, 146, 127,  70,  30,  45,  36, 215, 175,    ,    ,    , ...
    /// ...,   ,  30,    ,    ,    ,    ,    ,    ,    , 207, 246,  14,    ,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,  55, 251, 169,   1,    ,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,  11, 215, 232,  20,    ,    ,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,  20, 190, 250,  61,    ,    ,    ,    , ...
    /// ...,   ,    ,    ,    ,  24, 118, 206, 254, 248, 142, 108,  18,    ,    , ...
    /// ...,   ,    ,    ,  63, 223, 254, 254, 254, 254, 254, 254, 209,    ,    , ...
    /// ...,   ,    ,    ,  52, 174, 129,  95,  16,  16,  16, 106, 249, 125,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,    ,    ,    , 179, 239,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,    ,    ,    ,  80, 239,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,    ,    ,    ,  80, 244,  20, ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,    ,    ,    , 100, 239,    , ...
    /// ...,   ,    ,    ,    ,    ,    ,    ,    ,    ,    ,    , 234, 239,    , ...
    /// ...,  4, 140,   5,    ,    ,    ,    ,    ,    ,   3, 150, 254, 129,    , ...
    /// ..., 64, 254, 181,  38,    ,    ,    ,    ,  34, 188, 254, 209,  20,    , ...
    /// ..., 12, 226, 255, 223,  88,  68, 128, 157, 242, 254, 207,  23,    ,    , ...
    /// ...,   ,  45, 210, 254, 254, 254, 254, 255, 254, 187,  49,    ,    ,    , ...
    /// ...,   ,    ,  41, 129, 239, 229, 179,  91,  16,   3,    ,    ,    ,    , ...
    /// ...
    const TEST_IMG: [u8; 784] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 12, 56, 140, 126, 175, 200, 96, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 35, 166, 238, 254, 246, 242, 253, 246, 254, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 184, 182, 146, 127, 70, 30, 45, 36, 215, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 207, 246, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 251, 169, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 215, 232, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 190, 250, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 118, 206, 254, 248, 142, 108, 18, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 223, 254, 254, 254, 254, 254, 254, 209, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 174, 129, 95, 16, 16, 16, 106, 249, 125, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 239, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 239, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 244, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 239, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 239, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 140, 5, 0, 0, 0, 0, 0, 0, 3, 150, 254, 129, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 254, 181, 38, 0, 0, 0, 0, 34, 188, 254, 209, 20, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 226, 255, 223, 88, 68, 128, 157, 242, 254, 207, 23, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 210, 254, 254, 254, 254, 255, 254, 187,
        49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 129, 239, 229, 179, 91,
        16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ];
    const TEST_LBL: u8 = 3;
}
