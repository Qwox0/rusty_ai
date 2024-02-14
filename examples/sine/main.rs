#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use const_tensor::{Tensor, Vector};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use rusty_ai::{
    initializer::PytorchDefault,
    loss_function::SquaredError,
    nn::{Pair, TestResult},
    optimizer::sgd::SGD,
    trainer::Trainable,
    NNBuilder, Norm, NN,
};
use std::{fmt::Display, fs::File, io::Write, ops::Range, path::Path};

fn get_out_js_path() -> &'static str {
    if Path::new("./index.js").exists() {
        "./out.js"
    } else if Path::new("./examples/sine/index.js").exists() {
        "./examples/sine/out.js"
    } else {
        panic!("Can't find correct path for out.js")
    }
}

type X = f32;

pub fn main() {
    let out_js_path = get_out_js_path();
    println!("Generation output file: {}", out_js_path);
    std::fs::write(out_js_path, "").expect("could write out.js file");
    let mut js_file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(out_js_path)
        .expect("could open file");

    let mut rng = StdRng::seed_from_u64(69420);
    let mut ai = NNBuilder::default()
        .normal_precision()
        .rng(rng)
        .input_shape::<[(); 1]>()
        .layer::<20>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<20>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<20>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<1>(PytorchDefault, PytorchDefault)
        .build()
        .to_trainer()
        .loss_function(SquaredError)
        .optimizer(SGD { learning_rate: 0.01, ..SGD::default() })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 1000;
    const DATA_RANGE: Range<X> = -10.0..10.0;

    let training_data = Uniform::from(DATA_RANGE)
        .sample_iter(rand::thread_rng())
        .take(1000)
        .map(|x| Pair::new(Tensor::new([x]), Tensor::new([x.sin()])))
        .collect::<Vec<_>>();
    let test_data = Uniform::from(DATA_RANGE)
        .sample_iter(rand::thread_rng())
        .take(30)
        .map(|x| Pair::new(Vector::new([x]), Vector::new([x.sin()])))
        .collect::<Vec<_>>();

    writeln!(
        js_file,
        "let data = {{ x: '{}', y: '{}' }};",
        stringify_arr(test_data.iter().map(|p| p.get_input()[0].val())),
        stringify_arr(test_data.iter().map(|p| p.get_expected_output()[0].val()))
    )
    .unwrap();

    let mut js_res_vars = vec![];

    let (outputs, losses): (Vec<_>, Vec<_>) = ai.test_batch(&test_data).map(Into::into).unzip();
    let error = losses.into_iter().sum::<X>();
    println!("epoch: {:>4}, loss: {:<20}", 0, error);
    export_res(&mut js_file, &mut js_res_vars, 0, outputs, error);

    for epoch in 1..=EPOCHS {
        ai.train_rayon(&training_data);

        if epoch % 100 == 0 {
            let (outputs, losses): (Vec<_>, Vec<_>) =
                ai.test_batch(&test_data).map(Into::into).unzip();
            let error = losses.into_iter().sum::<X>();
            println!("epoch: {:>4}, loss: {:<20}", epoch, error);
            export_res(&mut js_file, &mut js_res_vars, epoch, outputs, error);
        }
    }

    writeln!(js_file, "let generations = [ {} ];", js_res_vars.join(", ")).unwrap();
}

fn export_res(
    js_file: &mut File,
    js_res_vars: &mut Vec<String>,
    epoch: usize,
    outputs: Vec<Vector<X, 1>>,
    error_sum: X,
) {
    let js_var_name = format!("gen{epoch}_result");
    writeln!(
        js_file,
        "let {js_var_name} = {{ gen: {epoch}, error: '{}', outputs: {} }};",
        error_sum,
        stringify_arr(outputs.into_iter().map(|out| out[0].val()))
    )
    .unwrap();
    js_res_vars.push(js_var_name);
}

fn stringify_arr(iter: impl Iterator<Item = impl Display>) -> String {
    format!("[{}]", iter.map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {

    #[test]
    fn seeded_test() {
        let mut rng = StdRng::seed_from_u64(69420);
        let mut ai = NNBuilder::default()
        .element_type::<X>()
        .rng(&mut rng)
        .input_shape::<[();1]>()
        .layer::<20>( PytorchDefault, PytorchDefault)
        .relu()
        .layer::<20>( PytorchDefault, PytorchDefault)
        .relu()
        .layer::<20>( PytorchDefault, PytorchDefault)
        .relu()
        .layer::<1>( PytorchDefault, PytorchDefault)
        .build()
        .to_trainer()
        .loss_function(SquaredError)
        .optimizer(SGD { learning_rate: 0.01, ..SGD::default() })
        //.optimizer(SGD::default())
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

        const EPOCHS: usize = 1000;
        const DATA_RANGE: Range<X> = -10.0..10.0;

        let training_data: Vec<_> = Uniform::from(DATA_RANGE)
            .sample_iter(&mut rng)
            .take(1000)
            .map(|input| Pair::new(Vector::new([input]), Vector::new([input.sin()])))
            .collect();
        let test_data: Vec<_> = Uniform::from(DATA_RANGE)
            .sample_iter(&mut rng)
            .take(30)
            .map(|input| Pair::new(Vector::new([input]), Vector::new([input.sin()])))
            .collect();

        let loss: X = ai.test_batch(test_data.iter()).map(|r| r.get_loss()).sum();
        println!("epoch: {:>4}, loss: {:<20}", 0, loss);

        for epoch in 1..=EPOCHS {
            ai.train_single_thread(&training_data);

            if epoch % 100 == 0 {
                let loss: X = ai.test_batch(test_data.iter()).map(|r| r.get_loss()).sum();
                println!("epoch: {:>4}, loss: {:<20}", epoch, loss);
            }
        }
    }
}
