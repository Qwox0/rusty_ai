#![allow(unused)]
mod args;

use clap::Parser;
use rand::Rng;
use rusty_ai::{
    activation_function::ActivationFunction,
    builder::NeuralNetworkBuilder,
    data_list::DataList,
    export::{ExportToJs, ExportedVariables},
    export_to_js,
    optimizer::OptimizerDispatch,
    results::{TestsResult, TrainingsResult},
};

const JS_OUTPUT_FILE: &str = "./out.js";
const INPUTS: usize = 1;
const TRAINING_RANGE: std::ops::RangeInclusive<f64> = -10.0..=10.0;

#[inline(always)]
fn target_fn(x: f64) -> f64 {
    x.sin()
}

fn main() {
    let args = args::Args::parse();

    // remove content
    std::fs::write(JS_OUTPUT_FILE, "").expect("could write out.js file");
    let mut js_file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(JS_OUTPUT_FILE)
        .expect("could open file");
    let mut result_names = ExportedVariables::new("generations");

    // create ai
    let relu = ActivationFunction::default_leaky_relu();
    let mut ai = NeuralNetworkBuilder::new()
        .input_layer::<1>()
        //.hidden_layers(&[3, 5, 3], relu)
        .hidden_layers(&[10, 10, 10], relu)
        .output_layer::<1>(ActivationFunction::Identity)
        .optimizer(OptimizerDispatch::default_gradient_descent())
        .build();

    println!("START: {}", ai);

    // create training data

    let training_data = DataList::random_simple(args.training_count, -10.0..10.0, target_fn);

    let (training_x, training_y): (Vec<f64>, Vec<f64>) =
        training_data.clone().into_iter_tuple_simple().unzip();
    training_x.export_to_js(&mut js_file, "training_x");
    training_y.export_to_js(&mut js_file, "training_y");

    let no_training_result = ai.test(training_data.iter());
    no_training_result.export_to_js(&mut js_file, "gen0_result");
    result_names.push("gen0_result").export(&mut js_file);

    // train

    ai.train(&training_data, 10, 10000, |ai, epoch| {
        if epoch % 10000 == 0 {
            println!("epoch: {}", epoch);
            let test_res = ai.test(training_data.iter());
            let res_name = format!("gen{}_result", epoch);
            test_res.export_to_js(&mut js_file, &res_name);
            result_names.push(&res_name).export(&mut js_file);
        }
    });

    println!("TRAINED: {}", ai);
    /*
    for epoch in 1..3 {
        ai.train(DataList::random_simple(5, target_fn).iter());
        if epoch % 1 == 0 {
            let no_training_result = ai.test(training_data.iter());
            let res_name = format!("gen{}_result", epoch);
            no_training_result.export_to_js(&mut js_file, &res_name);
            result_names.push(&res_name).export(&mut js_file);
        }
    }
    */

    /*
    let training_x: Vec<[f64; 1]> = training_x.into_iter().map(|x| [x; 1]).collect();
    let training_y = training_y.into_iter().map(|y| [y; 1]).collect();

    // train ai with training data
    let no_training_result = ai.test2(&training_x, &training_y);
    no_training_result.export_to_js(&mut js_file, "gen0_result");
    result_names.push("gen0_result").export(&mut js_file);

    dbg!(no_training_result);

    //let a = ai.train(training_pairs);

    let out = ai.propagate_many(&training_x);

    */
    /*
    let iter0_training_y: Vec<f64> = training_x
        .iter()
        .map(|input| ai.calculate(vec![*input]))
        .map(|out| out[0])
        .collect();

    let trainings_results = vec![(0usize, iter0_training_y.clone()), (1, iter0_training_y)];
    let result_str = trainings_results
        .iter()
        .map(|(gen, y)| format!("{{ gen: {gen}, output: {y:?} }}"))
        .collect::<Vec<_>>()
        .join(", ");

    // remove content
    std::fs::write(JS_OUTPUT_FILE, "").expect("could write out.js file");

    export_to_js!(&mut js_file => iterations = format!("[{}]", result_str));
    */
}
