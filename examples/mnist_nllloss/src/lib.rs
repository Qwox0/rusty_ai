#![feature(int_roundings)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(test)]

use rusty_ai::prelude::*;
use std::{cmp::Ordering, ops::Range, path::PathBuf, time::Instant};

#[cfg(not(target_os = "windows"))]
const NL: &'static str = "\n";
#[cfg(target_os = "windows")]
const NL: &'static str = "\r\n";

const IMAGE_SIDE: usize = 28;
const IMAGE_SIZE: usize = IMAGE_SIDE * IMAGE_SIDE;
const OUTPUTS: usize = 10;

pub fn get_data() -> load_mnist::Mnist {
    let exe_path = std::env::current_exe().expect("could get path of executable");
    let exe_dir = exe_path.parent().unwrap_or(&exe_path).display();
    let data_path = PathBuf::from(format!("{}/../../../examples/mnist/mnist-raw/", exe_dir));
    let data_path = data_path.as_os_str().to_str().expect("could convert path to &str");

    load_mnist::MnistBuilder::new()
        .base_path(data_path)
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize()
}

const NORMALIZE_MEAN: f64 = 0.5;
const NORMALIZE_STD: f64 = 0.5;
fn transform(img_vec: Vec<u8>, lbl_vec: Vec<u8>) -> PairList<IMAGE_SIZE, usize> {
    img_vec
        .into_iter()
        .map(|x| ((x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD)
        .array_chunks()
        .zip(lbl_vec.into_iter().map(usize::from))
        .collect()
}

fn setup_ai() -> NNTrainer<IMAGE_SIZE, OUTPUTS, NLLLoss, SGD_> {
    NNBuilder::default()
        .input::<IMAGE_SIZE>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::ReLU)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(OUTPUTS, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .log_softmax()
        .build::<OUTPUTS>()
        .to_trainable_builder()
        .loss_function(NLLLoss)
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

    const EPOCHS: usize = 5;
    const BATCH_SIZE: usize = 64;
    let batch_num = training_data.len().div_ceil(BATCH_SIZE);

    println!("\nTraining:");
    let start = Instant::now();

    for e in 0..EPOCHS {
        let mut running_loss = 0.0;
        for batch in training_data.chunks(BATCH_SIZE) {
            PairPropagation::new(&ai, batch);
            ai.training_step(batch);
            running_loss += 1.0;
        }
        // shuffle data after one full iteration
        training_data.shuffle();

        let training_loss = running_loss / batch_num as f64;

        println!("Epoch {} - Training loss: {}", e, training_loss); //, running_loss / len(trainloader));
        let secs = start.elapsed().as_secs();
        println!("Training Time: {} min {} s", secs / 60, secs % 60);
    }

    println!("\nTest:");
    for test in test_data.iter().take(3) {
        print_image(test, -1.0..1.0);
        let output = ai.propagate(&test.input);
        println!("output: {:?}", output.0);
        let propab = output.0.iter().copied().map(f64::exp).collect::<Vec<_>>();
        let guess = propab
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(Ordering::Less))
            .unwrap()
            .0;
        println!("propab: {:?}; guess: {}", propab, guess);
        let error = ai.get_loss_function().propagate_arr(&output.0, &test.expected_output);
        println!("error: {}", error);
        assert!(error < 0.2);
    }
}

pub fn print_image(pair: &Pair<IMAGE_SIZE, usize>, val_range: Range<f64>) {
    println!("{} label: {}", image_to_string(&pair.input, val_range), pair.expected_output);
}

/// each pixel has a size of 2x1 (`XX`).
pub fn image_to_string(image: &[f64; IMAGE_SIZE], val_range: Range<f64>) -> String {
    let v_border = "─".repeat(IMAGE_SIDE * 2);
    let v_border = v_border.as_str();

    let border_bytes = (IMAGE_SIDE * 6 + 4) * 3;
    let newline_bytes = (IMAGE_SIDE + 1) * NL.len();
    let mut buf = String::with_capacity(IMAGE_SIZE * 2 + border_bytes + newline_bytes);

    let range_width = val_range.end - val_range.start;
    buf.push('┌');
    buf.push_str(v_border);
    buf.push('┐');
    buf.push_str(NL);
    #[allow(illegal_floating_point_literal_pattern)]
    for line in image
        .iter()
        .map(|x| (x - val_range.start) / range_width)
        .map(|x| match x {
            ..=0.0751 => ' ',
            ..=0.0829 => '`',
            ..=0.0848 => '.',
            ..=0.1227 => '-',
            ..=0.1403 => '\'',
            ..=0.1559 => ':',
            ..=0.185 => '_',
            ..=0.2183 => ',',
            ..=0.2417 => '^',
            ..=0.2571 => '=',
            ..=0.2852 => ';',
            ..=0.2902 => '>',
            ..=0.2919 => '<',
            ..=0.3099 => '+',
            ..=0.3192 => '!',
            ..=0.3232 => 'r',
            ..=0.3294 => 'c',
            ..=0.3384 => '*',
            ..=0.3609 => '/',
            ..=0.3619 => 'z',
            ..=0.3667 => '?',
            ..=0.3737 => 's',
            ..=0.3747 => 'L',
            ..=0.3838 => 'T',
            ..=0.3921 => 'v',
            ..=0.396 => ')',
            ..=0.3984 => 'J',
            ..=0.3993 => '7',
            ..=0.4075 => '(',
            ..=0.4091 => '|',
            ..=0.4101 => 'F',
            ..=0.42 => 'i',
            ..=0.423 => '{',
            ..=0.4247 => 'C',
            ..=0.4274 => '}',
            ..=0.4293 => 'f',
            ..=0.4328 => 'I',
            ..=0.4382 => '3',
            ..=0.4385 => '1',
            ..=0.442 => 't',
            ..=0.4473 => 'l',
            ..=0.4477 => 'u',
            ..=0.4503 => '[',
            ..=0.4562 => 'n',
            ..=0.458 => 'e',
            ..=0.461 => 'o',
            ..=0.4638 => 'Z',
            ..=0.4667 => '5',
            ..=0.4686 => 'Y',
            ..=0.4693 => 'x',
            ..=0.4703 => 'j',
            ..=0.4833 => 'y',
            ..=0.4881 => 'a',
            ..=0.4944 => ']',
            ..=0.4953 => '2',
            ..=0.4992 => 'E',
            ..=0.5509 => 'S',
            ..=0.5567 => 'w',
            ..=0.5569 => 'q',
            ..=0.5591 => 'k',
            ..=0.5602 => 'P',
            //..=0.5602 => '6',
            ..=0.565 => 'h',
            ..=0.5776 => '9',
            ..=0.5777 => 'd',
            ..=0.5818 => '4',
            ..=0.587 => 'V',
            ..=0.5972 => 'p',
            ..=0.5999 => 'O',
            ..=0.6043 => 'G',
            ..=0.6049 => 'b',
            ..=0.6093 => 'U',
            ..=0.6099 => 'A',
            ..=0.6465 => 'K',
            ..=0.6561 => 'X',
            ..=0.6595 => 'H',
            ..=0.6631 => 'm',
            ..=0.6714 => '8',
            ..=0.6759 => 'R',
            ..=0.6809 => 'D',
            ..=0.6816 => '#',
            ..=0.6925 => '$',
            ..=0.7039 => 'B',
            ..=0.7086 => 'g',
            ..=0.7235 => '0',
            ..=0.7302 => 'M',
            ..=0.7332 => 'N',
            ..=0.7602 => 'W',
            ..=0.7834 => 'Q',
            ..=0.8037 => '%',
            //..=0.9999 => '&',
            ..=0.95 => '&',
            _ => '@',
        })
        .array_chunks::<IMAGE_SIDE>()
    {
        buf.push('│');
        for char in line {
            buf.push(char);
            buf.push(char);
        }
        buf.push('│');
        buf.push_str(NL);
    }
    buf.push('└');
    buf.push_str(v_border);
    buf.push('┘');
    assert_eq!(buf.capacity(), buf.len()); // not necessary
    buf
}
