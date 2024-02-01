use rusty_ai::{Initializer, NNBuilder, NN};
use serde::de::DeserializeOwned;

const IN: usize = 2;
const OUT: usize = 3;

fn get_nn() -> impl NN<f32, [(); IN], [(); OUT]> + DeserializeOwned {
    NNBuilder::default()
        .default_rng()
        .input_shape::<[(); IN]>()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<6>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer::<OUT>(Initializer::PytorchDefault, Initializer::PytorchDefault)
        .sigmoid()
        .build()
}

fn main() -> Result<(), serde_json::Error> {
    let nn = get_nn();
    println!("{}\n", nn);

    let json = serde_json::to_string(&nn)?;
    println!("\nJSON: {}", json);

    let new_nn = nn.deserialize_hint(serde_json::from_str(&json)?);

    println!("{:?}", new_nn);
    assert_eq!(nn, new_nn);
    println!("\nAI and NEW_AI are equal!");

    Ok(())
}
