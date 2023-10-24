use rusty_ai::*;

fn main() -> Result<(), serde_json::Error> {
    const IN: usize = 2;
    const OUT: usize = 3;

    let ai = NNBuilder::default()
        .input::<IN>()
        .layer(6, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(6, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .relu()
        .layer(OUT, Initializer::PytorchDefault, Initializer::PytorchDefault)
        .sigmoid()
        .build::<OUT>();
    println!("AI: {}", ai);

    let json = serde_json::to_string(&ai)?;
    println!("\nJSON: {}", json);

    let new_ai: NeuralNetwork<IN, OUT> = serde_json::from_str(&json)?;
    println!("\nNEW_AI: {}", new_ai);

    assert_eq!(ai, new_ai);
    println!("\nAI and NEW_AI are equal!");

    Ok(())
}
