#![feature(test)]

use rusty_ai::prelude::*;
use std::{fmt::Display, fs::File, io::Write, path::Path};

fn get_out_js_path() -> &'static str {
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

    let mut ai = NeuralNetworkBuilder::default()
        .default_activation_function(ActivationFn::ReLU)
        .input::<1>()
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(1, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .identity()
        .build::<1>()
        .to_trainable_builder()
        .loss_function(SquaredError)
        .sgd(GradientDescent { learning_rate: 0.01 })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 1000;

    let training_data = DataBuilder::uniform(-2.0..2.0).build::<1>(1000).gen_pairs(|[x]| [x.sin()]);
    let test_data = DataBuilder::uniform(-2.0..2.0).build::<1>(30).gen_pairs(|[x]| [x.sin()]);
    let (x, y): (Vec<_>, Vec<_>) =
        test_data.iter().map(|pair| (pair.input[0], pair.expected_output[0])).unzip();

    writeln!(
        js_file,
        "let data = {{ x: '{}', y: '{}' }};",
        stringify_arr(x.iter()),
        stringify_arr(y.iter())
    )
    .unwrap();

    let mut js_res_vars = vec![];

    let res = ai.test(test_data.iter());
    println!("epoch: {:>4}, loss: {:<20}", 0, res.error);
    export_res(&mut js_file, &mut js_res_vars, 0, res);

    ai.full_train(&training_data, EPOCHS, |epoch, ai| {
        if epoch % 100 == 0 {
            let res = ai.test(&SquaredError, test_data.iter());
            println!("epoch: {:>4}, loss: {:<20}", epoch, res.error);
            export_res(&mut js_file, &mut js_res_vars, epoch, res);
        }
    });

    writeln!(js_file, "let generations = [ {} ];", js_res_vars.join(", ")).unwrap();
}

fn export_res(
    js_file: &mut File,
    js_res_vars: &mut Vec<String>,
    epoch: usize,
    res: TestsResult<1>,
) {
    let js_var_name = format!("gen{epoch}_result");
    writeln!(
        js_file,
        "let {js_var_name} = {{ gen: {epoch}, error: '{}', outputs: {} }};",
        res.error,
        stringify_arr(res.outputs.into_iter().map(|out| out.0[0]))
    )
    .unwrap();
    js_res_vars.push(js_var_name);
}

fn stringify_arr(iter: impl Iterator<Item = impl Display>) -> String {
    format!("[{}]", iter.map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}
