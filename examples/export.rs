#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rusty_ai::{Initializer, NNBuilder};

fn main() -> Result<(), serde_json::Error> {
    const IN: usize = 2;
    const OUT: usize = 3;

    let nn = NNBuilder::default()
        .default_rng()
        .input_shape::<[(); IN]>()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<OUT>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .sigmoid()
        .build();
    println!("NN: {}", nn);

    /*
    let json = serde_json::to_string(&nn)?;
    println!("\nJSON: {}", json);

    let new_ai: NN<f32, Vector<f32, IN>, Vector<f32, OUT>, ()> = serde_json::from_str(&json)?;
    println!("\nNEW_AI: {}", new_ai);

    assert_eq!(ai, new_ai);
    println!("\nAI and NEW_AI are equal!");
    */

    Ok(())
}
