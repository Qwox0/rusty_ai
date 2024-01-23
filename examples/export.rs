#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rusty_ai::{
    nn::{Linear, ReLU, Sigmoid},
    Initializer, NN,
};

const IN: usize = 2;
const OUT: usize = 3;

fn main() -> Result<(), serde_json::Error> {
    pub type MyNN = NN<
        f32,
        [(); 2],
        [(); 3],
        Sigmoid<Linear<f32, 6, 3, ReLU<Linear<f32, 6, 6, ReLU<Linear<f32, 2, 6, ()>>>>>>,
    >;

    let nn: MyNN = NN::builder()
        .default_rng()
        .input_shape::<[(); IN]>()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<OUT>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .sigmoid()
        .build();
    println!("{}\n", nn);

    let json = serde_json::to_string(&nn)?;
    println!("\nJSON: {}", json);

    let new_nn: MyNN = serde_json::from_str(&json)?;

    assert_eq!(nn, new_nn);
    println!("\nAI and NEW_AI are equal!");

    Ok(())
}
