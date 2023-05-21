// Answer to "Approximating the sine function with a neural network"
// https://stackoverflow.com/a/45197060/15200927

use rusty_ai::export::{ExportToJs, ExportedVariables};
use rusty_ai::prelude::*;
use std::f64::consts::PI;

const JS_OUTPUT_FILE: &str = "./out.js";

fn main() {
    std::fs::write(JS_OUTPUT_FILE, "").expect("could write out.js file");
    let mut js_file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(JS_OUTPUT_FILE)
        .expect("could open file");
    let mut result_names = ExportedVariables::new("generations");

    fn vec<const W: usize, const H: usize>(arr: [[f64; W]; H]) -> Vec<Vec<f64>> {
        arr.into_iter().map(Vec::from).collect()
    }

    let relu = ActivationFn::default_relu();

    let mut ai = NeuralNetworkBuilder::new()
        .input_layer::<1>()
        .hidden_layers_random(&[100, 100, 100], relu)
        .output_layer::<1>(ActivationFn::Identity)
        .error_function(ErrorFunction::SquaredError)
        .gradient_descent_optimizer(GradientDescent {
            learning_rate: 0.01,
        })
        //.adam_optimizer(Adam::with_learning_rate(0.01))
        .build();

    let data = DataList::random_simple(500, -PI..PI, f64::sin);
    data.export_to_js(&mut js_file, "data");

    let no_training_result = ai.test_propagate(data.iter());
    no_training_result.export_to_js(&mut js_file, "gen0_result");
    result_names.push("gen0_result").export(&mut js_file);

    let test_res = ai.test_propagate(data.iter());
    println!("epoch: 0, error: {}", test_res.error);

    ai.train(&data, 32, 6000, |epoch, ai| {
        if epoch % 100 == 0 {
            let test_res = ai.test_propagate(data.iter());
            println!("epoch: {}, error: {}", epoch, test_res.error);
            let res_name = format!("gen{}_result", epoch);
            test_res.export_to_js(&mut js_file, &res_name);
            result_names.push(&res_name).export(&mut js_file);
        }
    });

    //println!("TRAINED: {}", ai);
    println!("done");
}
