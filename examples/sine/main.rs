#![feature(test)]

use rusty_ai::{data::DataBuilder, loss_function::SquaredError, optimizer::sgd::SGD, *};
use std::{
    fmt::Display,
    fs::File,
    io::Write,
    ops::Range,
    path::{Path, PathBuf},
};

fn get_out_js_path() -> &'static str {
    let a = env!("CARGO_MANIFEST_DIR");
    println!("{:?}", PathBuf::from(a));
    println!("{:?}", Path::new(".").canonicalize());
    if Path::new("./index.js").exists() {
        "./out.js"
    } else if Path::new("./examples/sine/index.js").exists() {
        "./examples/sine/out.js"
    } else {
        panic!("Can't find correct path for out.js")
    }
}

pub fn main() {
    let out_js_path = get_out_js_path();
    println!("Generation output file: {}", out_js_path);
    std::fs::write(out_js_path, "").expect("could write out.js file");
    let mut js_file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(out_js_path)
        .expect("could open file");

    let mut ai = NNBuilder::default()
        .default_activation_function(ActivationFn::ReLU)
        .input::<1>()
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(1, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .identity()
        .build::<1>()
        .to_trainer()
        .loss_function(SquaredError)
        .optimizer(SGD { learning_rate: 0.01, ..SGD::default() })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 1000;
    const DATA_RANGE: Range<f64> = -10.0..10.0;

    let training_data =
        DataBuilder::uniform(DATA_RANGE).build::<1>(1000).gen_pairs(|[x]| [x.sin()]);
    let test_data = DataBuilder::uniform(DATA_RANGE).build::<1>(30).gen_pairs(|[x]| [x.sin()]);
    let (x, y): (Vec<_>, Vec<_>) = test_data
        .iter()
        .map(|(input, expected_output)| (input[0], expected_output[0]))
        .unzip();

    writeln!(
        js_file,
        "let data = {{ x: '{}', y: '{}' }};",
        stringify_arr(x.iter()),
        stringify_arr(y.iter())
    )
    .unwrap();

    let mut js_res_vars = vec![];

    let (outputs, losses): (Vec<_>, Vec<_>) = ai.test_batch(test_data.iter()).unzip();
    let error = losses.into_iter().sum::<f64>();
    println!("epoch: {:>4}, loss: {:<20}", 0, error);
    export_res(&mut js_file, &mut js_res_vars, 0, outputs, error);

    for epoch in 0..EPOCHS {
        ai.train(&training_data).execute();

        if epoch % 100 == 0 {
            let (outputs, losses): (Vec<_>, Vec<_>) = ai.test_batch(test_data.iter()).unzip();
            let error = losses.into_iter().sum::<f64>();
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
    outputs: Vec<[f64; 1]>,
    error_sum: f64,
) {
    let js_var_name = format!("gen{epoch}_result");
    writeln!(
        js_file,
        "let {js_var_name} = {{ gen: {epoch}, error: '{}', outputs: {} }};",
        error_sum,
        stringify_arr(outputs.into_iter().map(|out| out[0]))
    )
    .unwrap();
    js_res_vars.push(js_var_name);
}

fn stringify_arr(iter: impl Iterator<Item = impl Display>) -> String {
    format!("[{}]", iter.map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}
