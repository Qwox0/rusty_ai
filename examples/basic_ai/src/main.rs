use clap::Parser;
use rand::Rng;
use rusty_ai::{builder::NeuralNetworkBuilder, neural_network::NeuralNetwork};
use std::fmt::Debug;

const JS_OUTPUT_FILE: &str = "./out.js";

#[derive(Debug, clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short = 'c', default_value_t = 10)]
    training_count: usize,
}

fn main() {
    let args = Args::parse();

    let mut rng = rand::thread_rng();
    let training_x: Vec<f64> = (0..args.training_count)
        .map(|_| rng.gen_range(-10.0..=10.0))
        .collect();
    let training_y: Vec<f64> = training_x
        .clone()
        .into_iter()
        .map(|x: f64| f64::sin(x))
        .collect();

    //let training_pairs: Vec<(&f64, &f64)> = training_x.iter().zip(training_y.iter()).collect();
    use rusty_ai::activation_function::ActivationFunction::*;
    let ai = NeuralNetworkBuilder::input_layer(1)
        .hidden_layer(2, ReLU2)
        .hidden_layers(&[3, 2], ReLU2)
        .output_layer(1, ReLU2)
        .build();

    println!("ai: {}", ai);

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

    macro_rules! export_to_js {
        ( $js_file:expr => $( $var:ident = $val:expr ),* ) => {{
            use std::io::prelude::Write;
            let file: &mut ::std::fs::File = $js_file;
            $(
                writeln!(file, "let {} = {};", stringify!($var), $val).expect("could write to file");
            )*
        }};
        ( $js_file:expr => $( $var:ident ),* ) => {
            export_to_js!($js_file => $( $var = format!("'{:?}'", $var)),*)
        };
    }

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(JS_OUTPUT_FILE)
        .expect("could open file");
    export_to_js!(&mut file => training_x, training_y);
    export_to_js!(&mut file => iterations = format!("[{}]", result_str));

    let input = vec![0.0];
    println!("input: {:?} -> output: {:?}", &input, ai.calculate_ref(&input));
}
