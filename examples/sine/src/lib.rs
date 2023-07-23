#![feature(test)]

use rusty_ai::{
    export::{ExportToJs, ExportedVariables},
    prelude::*,
};
use std::{fs::File, io::Write, path::Path};

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
    let mut result_names = ExportedVariables::new("generations");

    let mut ai = NeuralNetworkBuilder::default()
        .default_activation_function(ActivationFn::ReLU(0.0))
        .input::<1>()
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(20, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .layer(1, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .activation_function(ActivationFn::Identity)
        .error_function(ErrorFunction::SquaredError)
        .build::<1>()
        .to_trainable_builder()
        .sgd(GradientDescent { learning_rate: 0.01 })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    const EPOCHS: usize = 1000;

    let training_data = DataBuilder::uniform(-2.0..2.0).build::<1>(1000).gen_pairs(|[x]| [x.sin()]);
    let test_data = DataBuilder::uniform(-2.0..2.0).build::<1>(30).gen_pairs(|[x]| [x.sin()]);
    test_data.export_to_js(&mut js_file, "data");

    let mut js_res_vars = vec![];

    let res = ai.test_propagate(test_data.iter());
    println!("epoch: {:>4}, loss: {:<20}", 0, res.error);
    export_res(&mut js_file, &mut js_res_vars, 0, res);

    ai.full_train(&training_data, EPOCHS, |epoch, ai| {
        if epoch % 100 == 0 {
            let res = ai.test_propagate(test_data.iter());
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
        "let {js_var_name} = {{ gen: {epoch}, error: '{}', outputs: [{}] }};",
        res.error,
        res.outputs.into_iter().map(|out| format!("{}", out.0[0])).collect::<Vec<_>>().join(",")
    )
    .unwrap();
    js_res_vars.push(js_var_name);

    //let gen400_result = { error: '10.053866127675557', outputs: [0.14807035647034178,
    // -0.7917905689621438, -0.49526150319343965, -0.8513780207835094, -0.17993831721165388,
    // 0.14807035647034178, -0.8548794817372863, -0.4830072998613164, 1.37948929184945,
    // 1.4769100643706403, -0.15053645814796968, 0.14807035647034178, 0.14807035647034178,
    // 0.8783821697585126, -0.5378436537509715, 0.017570767374352772, 0.11982417131207236,
    // -0.861017252909277, 0.8054305691075897, 0.14807035647034178, 0.14807035647034178,
    // 0.26032399314725874, 0.14807035647034178, 0.14807035647034178, 0.14807035647034178,
    // 0.14807035647034178, 0.14807035647034178, 0.14807035647034178, 0.6078028467185074,
    // 0.14807035647034178] };
}
