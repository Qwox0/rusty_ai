#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(test)]

use rusty_ai::prelude::*;
use std::{iter::once, ops::Range, path::PathBuf, time::Instant};

#[cfg(not(target_os = "windows"))]
const NL: &'static str = "\n";
#[cfg(target_os = "windows")]
const NL: &'static str = "\r\n";

const IMAGE_SIDE: usize = 28;
const IMAGE_SIZE: usize = IMAGE_SIDE * IMAGE_SIDE;
const OUTPUTS: usize = 10;

const NORMALIZE_MEAN: f64 = 0.5;
const NORMALIZE_STD: f64 = 0.5;
fn transform(img_vec: Vec<u8>, lbl_vec: Vec<u8>) -> PairList<IMAGE_SIZE, OUTPUTS> {
    img_vec
        .into_iter()
        .map(|x| ((x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD)
        .array_chunks()
        //.zip(lbl_vec.into_iter().map(|x| [x as f64; OUTPUTS])) // TODO: remove OUTPUTS
        .zip(lbl_vec.into_iter().map(|x| {
            let mut out = [0.0; OUTPUTS]; // TODO: remove OUTPUTS
            out[x as usize] = 1.0;
            out
        }))
        .collect()
}

pub fn main() {
    let exe_path = std::env::current_exe().expect("could get path of executable");
    let exe_dir = exe_path.parent().unwrap_or(&exe_path).display();
    let data_path = PathBuf::from(format!("{}/../../../examples/mnist/mnist-raw/", exe_dir));
    let data_path = data_path.as_os_str().to_str().expect("could convert path to &str");

    let load_mnist::Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } =
        load_mnist::MnistBuilder::new()
            .base_path(data_path)
            .label_format_digit()
            .training_set_length(60_000)
            .validation_set_length(0)
            .test_set_length(10_000)
            .finalize();

    let mut training_data = transform(trn_img, trn_lbl);
    let test_data = transform(tst_img, tst_lbl);

    assert_eq!(training_data.len(), 60_000);
    assert_eq!(test_data.len(), 10_000);

    // TODO: print_image(&training_data[0], -1.0..1.0);
    print_image_tmp(&training_data[50], -1.0..1.0);

    let mut ai = NeuralNetworkBuilder::default()
        .input::<IMAGE_SIZE>()
        .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu(0.0)
        .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu(0.0)
        .layer(10, Initializer::PytorchDefault, Initializer::PytorchDefault)
        //.activation_function(ActivationFn::LogSoftmax)
        .activation_function(ActivationFn::Sigmoid)
        .error_function(ErrorFunction::SquaredError)
        .build::<10>()
        .to_trainable_builder()
        .sgd(GradientDescent { learning_rate: 0.003 })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two) // ?
        .build();

    const EPOCHS: usize = 5;
    const BATCH_SIZE: usize = 64;

    let start = Instant::now();

    for e in 0..EPOCHS {
        for batch in training_data.chunks(BATCH_SIZE) {
            ai.training_step(batch)
        }
        // shuffle data after one full iteration
        training_data.shuffle();

        println!("Epoch {} - Training loss: ", e); //, running_loss / len(trainloader));
        let secs = start.elapsed().as_secs();
        println!("Training Time: {} min {} s", secs / 60, secs % 60);
    }

    for test in test_data.iter().take(3) {
        print_image_tmp(test, -1.0..1.0);
        let output = ai.propagate(&test.input);
        println!("output: {:?}", output.0);
        let errors = output
            .0
            .iter()
            .zip(&test.output)
            .map(|(o, e)| o - e)
            .map(|x| x * x)
            .collect::<Vec<_>>();
        println!("errors: {:?}", errors);
        let errors_sum = errors.iter().sum::<f64>();
        println!("errors sum: {}", errors_sum);
        assert!(errors_sum < 0.2);
    }

    /*
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

pub fn print_image_tmp(pair: &Pair<IMAGE_SIZE, 10>, val_range: Range<f64>) {
    println!("{} label: \n{:?}", image_to_string(&pair.input, val_range), pair.output);
}

pub fn print_image(pair: &Pair<IMAGE_SIZE, 1>, val_range: Range<f64>) {
    println!("{} label: {}", image_to_string(&pair.input, val_range), pair.output[0]);
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

pub mod tests {
    extern crate test;
    use super::*;
    use test::*;

    fn load_data() -> Pair<IMAGE_SIZE, 1> {
        Pair::new(
            TEST_IMG
                .iter()
                .map(|x| ((*x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD)
                .array_chunks()
                .next()
                .unwrap(),
            [TEST_LBL as f64],
        )
    }

    fn load_data_hack() -> Pair<IMAGE_SIZE, 10> {
        let Pair { input, output } = load_data();
        let idx = output[0] as usize;
        let mut output = [0.0; 10];
        output[idx] = 1.0;
        Pair::new(input, output)
    }

    fn setup_nn() -> TrainableNeuralNetwork<IMAGE_SIZE, 10> {
        NeuralNetworkBuilder::default()
            .input::<IMAGE_SIZE>()
            .layer(128, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .relu(0.0)
            .layer(64, Initializer::PytorchDefault, Initializer::PytorchDefault)
            .relu(0.0)
            .layer(10, Initializer::PytorchDefault, Initializer::PytorchDefault)
            //.activation_function(ActivationFn::LogSoftmax)
            .activation_function(ActivationFn::Sigmoid)
            .error_function(ErrorFunction::SquaredError)
            .build::<10>()
            .to_trainable_builder()
            .sgd(GradientDescent { learning_rate: 0.003 })
            .retain_gradient(true)
            .new_clip_gradient_norm(5.0, Norm::Two) // ?
            .build()
    }

    /// #[bench]
    pub fn test_propagate(b: &mut Bencher) {
        let ai = setup_nn();
        let p = load_data();

        b.iter(|| {
            black_box(ai.propagate(black_box(&p.input)));
        })
    }

    /// #[bench]
    pub fn test_train_hack(b: &mut Bencher) {
        let mut ai = setup_nn();
        let p = load_data_hack();

        b.iter(|| {
            ai.training_step(once(&p));
        })
    }

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
