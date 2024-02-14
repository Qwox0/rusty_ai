#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(int_roundings)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(test)]

use const_tensor::{Element, Multidimensional, Num, Vector};
use mnist_util::{get_mnist, get_mnist_with_len, image_to_string, Mnist};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rusty_ai::{
    initializer::PytorchDefault,
    loss_function::NLLLoss,
    nn::{NNBuilder, Pair},
    optimizer::sgd::SGD,
    trainer::Trainable,
    Norm, NN,
};
use std::{ops::Range, time::Instant};

const IMAGE_SIDE: usize = 28;
const IMAGE_SIZE: usize = IMAGE_SIDE * IMAGE_SIDE;
const OUTPUTS: usize = 10;

const NORMALIZE_MEAN: f64 = 0.5;
const NORMALIZE_STD: f64 = 0.5;
fn transform<X: Num>(img_vec: Vec<u8>, lbl_vec: Vec<u8>) -> Vec<Pair<X, [(); IMAGE_SIZE], usize>> {
    img_vec
        .into_iter()
        .map(|x| (((x as f64) / 256.0 - NORMALIZE_MEAN) / NORMALIZE_STD).cast())
        .array_chunks()
        .zip(lbl_vec.into_iter().map(usize::from))
        .map(|(input, eo)| Pair::new(Vector::new(input), eo))
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

    /*
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            */
    let mut rng = rand::thread_rng();

    let mut ai = NNBuilder::default()
        .double_precision()
        .rng(&mut rng)
        .input_shape::<[(); IMAGE_SIZE]>()
        .layer::<128>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<64>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<OUTPUTS>(PytorchDefault, PytorchDefault)
        .log_softmax()
        .build()
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
            let loss =
                ai.train_rayon_output(batch).map(|out| out.loss).sum::<f64>() / BATCH_SIZE as f64;
            running_loss += loss;
        }
        // shuffle data after one full iteration
        training_data.shuffle(&mut rng);

        let training_loss = running_loss / batch_num as f64;

        println!("Epoch {} - Training loss: {}", e, training_loss);
        let secs = start.elapsed().as_secs();
        println!("Training Time: {} min {} s", secs / 60, secs % 60);
    }

    println!("\nTest:");
    for pair in test_data.iter().take(3) {
        print_image(pair, -1.0..1.0);
        let (out, loss) = ai.test(pair).into_tuple();
        println!("output: {:?}", out);
        let propab = out.iter_elem().copied().map(f64::exp).collect::<Vec<_>>();
        let guess = propab.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        println!("propab: {:?}; guess: {}", propab, guess);
        println!("error: {}", loss);
        assert!(loss < 0.2);
    }
}

pub fn print_image<X: Num>(pair: &Pair<X, [(); IMAGE_SIZE], usize>, val_range: Range<X>) {
    println!(
        "{} label: {}",
        image_to_string::<X, IMAGE_SIDE, IMAGE_SIZE>(pair.get_input().as_arr(), val_range),
        pair.get_expected_output()
    );
}

fn seeded_test() {
    type X = f32;

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = get_mnist_with_len(500, 10);

    let mut training_data = transform::<X>(trn_img, trn_lbl);
    let test_data = transform::<X>(tst_img, tst_lbl);

    println!("Example Image:");
    print_image(&training_data[50], -1.0..1.0);

    let mut rng = StdRng::seed_from_u64(69420);

    let mut ai = NNBuilder::default()
        .element_type::<X>()
        .rng(&mut rng)
        .input_shape::<[(); IMAGE_SIZE]>()
        .layer::<128>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<64>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<OUTPUTS>(PytorchDefault, PytorchDefault)
        .log_softmax()
        .build()
        .to_trainer()
        .loss_function(NLLLoss)
        .optimizer(SGD { learning_rate: 0.003, momentum: 0.9 })
        .retain_gradient(false)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 30;
    const BATCH_SIZE: usize = 64;
    let batch_num = training_data.len().div_ceil(BATCH_SIZE);

    println!("\nTraining:");
    let start = Instant::now();

    for e in 0..EPOCHS {
        let mut running_loss = 0.0;
        for batch in training_data.chunks(BATCH_SIZE) {
            let loss = ai.train_rayon_output(batch).map(|out| out.loss).sum::<X>()
                / BATCH_SIZE as X;
            running_loss += loss;
        }
        // shuffle data after one full iteration
        training_data.shuffle(&mut rng);

        let training_loss = running_loss / batch_num as X;

        println!("Epoch {:>2} - Training loss: {}", e, training_loss);
        let secs = start.elapsed().as_secs();
        //println!("Training Time: {} min {} s", secs / 60, secs % 60);
    }

    println!("\nTest:");
    for pair in test_data.iter().take(3) {
        print_image(pair, -1.0..1.0);
        let (out, loss) = ai.test(pair).into_tuple();
        println!("output: {:?}", out);
        let propab = out.iter_elem().copied().map(X::exp).collect::<Vec<_>>();
        let guess = propab.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        println!("propab: {:?}; guess: {}", propab, guess);
        println!("error: {}", loss);
        //assert!(loss < 0.2);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn seeded_test() {
        super::seeded_test();
    }
}
