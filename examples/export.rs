use rusty_ai::{initializer::PytorchDefault, NNBuilder, NN};
use serde::de::DeserializeOwned;

const IN: usize = 2;
const OUT: usize = 3;

fn get_nn() -> impl NN<f32, [(); IN], [(); OUT]> + DeserializeOwned {
    NNBuilder::default()
        .default_rng()
        .input_shape::<[(); IN]>()
        .layer::<6>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<6>(PytorchDefault, PytorchDefault)
        .relu()
        .layer::<OUT>(PytorchDefault, PytorchDefault)
        .sigmoid()
        .build()
}

fn main() -> Result<(), serde_json::Error> {
    let nn = get_nn();

    let json = serde_json::to_string(&nn)?;
    println!("JSON: {}", json);

    let new_nn = nn.deserialize_hint(serde_json::from_str(&json)?);

    assert_eq!(nn, new_nn);
    println!("\nNN and NEW_NN are equal!");

    Ok(())
}
