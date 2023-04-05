use rusty_ai::neural_network::NeuralNetwork;
use clap::Parser;
use rand::Rng;
use std::fmt::Debug;

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

    let ai = NeuralNetwork::new(&[1, 2, 3, 2, 1]);



    //println!("{:?}", ai);
    println!("ai: {}", ai);

    let iter0_training_y: Vec<f64> = training_x
        .iter()
        .map(|input| ai.calculate(&vec![*input]))
        .map(|out| out[0])
        .collect();

    macro_rules! export_to_js {
        ( $path:literal: $( $var:ident),* ) => {
            std::fs::write(
                $path,
                format!(
                    concat!( $("let ", stringify!($var), " = '{:?}';\n"),* ),
                    $($var),*
                )
                .as_bytes(),
            ).expect("could write out.js file");
        };
    }

    export_to_js!("./out.js": training_x, training_y, iter0_training_y);

    let input = vec![0.0];
    println!("input: {:?} -> output: {:?}", &input, ai.calculate(&input));
}
