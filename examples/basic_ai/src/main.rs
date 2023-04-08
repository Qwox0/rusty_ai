mod args;

use clap::Parser;
use rand::Rng;
use rusty_ai::{
    activation_function::ActivationFunction,
    builder::NeuralNetworkBuilder,
    export::{ExportToJs, ExportedVariables},
    export_to_js,
    results::{TestsResult, TrainingsResult},
};

const JS_OUTPUT_FILE: &str = "./out.js";
const INPUTS: usize = 1;
const TRAINING_RANGE: std::ops::RangeInclusive<f64> = -30.0..=30.0;

#[inline(always)]
fn target_fn(x: &f64) -> f64 {
    x.exp()
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
    let relu = ActivationFunction::default_relu();
    let mut ai = NeuralNetworkBuilder::new()
        .input_layer::<1>()
        .hidden_layers(&[2, 3, 2], relu)
        .output_layer::<1>(relu)
        .build();

    println!("ai: {}", ai);

    // create training data
    let mut rng = rand::thread_rng();
    let training_x: Vec<f64> = (0..args.training_count)
        .map(|_| rng.gen_range(TRAINING_RANGE))
        .collect();
    let training_y: Vec<f64> = training_x.iter().map(target_fn).collect();
    training_x.export_to_js(&mut js_file, "training_x"); // js doesn't need to know, that inputs are actually Vec<f64> instead of f64
    training_y.export_to_js(&mut js_file, "training_y");

    let training_x: Vec<[f64; 1]> = training_x.into_iter().map(|x| [x; 1]).collect();
    let training_y = training_y.into_iter().map(|y| [y; 1]).collect();

    // train ai with training data
    let no_training_result = ai.test2(&training_x, &training_y);
    no_training_result.export_to_js(&mut js_file, "gen0_result");
    result_names.push("gen0_result").export(&mut js_file);

    dbg!(no_training_result);

    //let a = ai.train(training_pairs);

    let out = ai.propagate_many(&training_x);

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
