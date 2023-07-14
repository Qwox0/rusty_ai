use rusty_ai::{export::ExportedVariables, prelude::*};

const JS_OUTPUT_FILE: &str = "./out.js";

#[test]
fn test() {
    main();
}

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
    /*
    [[0.2654893009262182],
    [0.13379694515812643]]
    biases:  [-0.3220212329837344, 0.839882040384166]
    weights:
    [[0.16532130582925397, 0.07010053781757337],
    [-0.5854114069916742, 0.6643306908932561]]
    biases:  [-0.35676113792377323, -0.06590416343319068]
    weights:
    [[-0.488072052229262, 0.3707835128103556],
    [0.14340687126247803, -0.06900138303413593]]
    biases:  [-0.5220346327969696, -0.47481103449761164]
    weights:
    [[0.14407180895539873, 0.1776979799751251]]
    biases:  [0.5294013892421578]
         */

    let w1 = Matrix::from_elements(vec([[0.2654893009262182], [0.13379694515812643]]));
    let b1 = LayerBias::OnePerNeuron([-0.3220212329837344, 0.839882040384166].to_vec());
    let w2 = Matrix::from_elements(vec([
        [0.16532130582925397, 0.07010053781757337],
        [-0.5854114069916742, 0.6643306908932561],
    ]));
    let b2 = LayerBias::OnePerNeuron([-0.35676113792377323, -0.06590416343319068].to_vec());
    let w3 = Matrix::from_elements(vec([
        [-0.488072052229262, 0.3707835128103556],
        [0.14340687126247803, -0.06900138303413593],
    ]));
    let b3 = LayerBias::OnePerNeuron([-0.5220346327969696, -0.47481103449761164].to_vec());
    let w4 = Matrix::from_elements(vec([[0.14407180895539873, 0.1776979799751251]]));
    let b4 = LayerBias::OnePerNeuron([0.5294013892421578].to_vec());

    let relu = ActivationFn::default_relu();

    let mut ai = NeuralNetworkBuilder::new()
        .input_layer::<1>()
        .hidden_layer(Layer::new(w1, b1, relu))
        .hidden_layer(Layer::new(w2, b2, relu))
        .hidden_layer(Layer::new(w3, b3, relu))
        .output_layer_unchecked(Layer::new(w4, b4, relu))
        //.hidden_layers(&[100, 100, 100], relu)
        //.output_layer::<1>(ActivationFn::Identity)
        .error_function(ErrorFunction::SquaredError)
        .gradient_descent_optimizer(GradientDescent {
            learning_rate: 0.01,
        })
        //.adam_optimizer(Adam::with_learning_rate(0.01))
        .build();

    println!("ai: {}", ai);

    //let data = DataList::random_simple(2000, -PI..PI, f64::sin);
    let data = DataList::from(vec![(2.0, 2.0f64.sin())]);
    println!("\ndata: {:?}\n", data);

    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 0, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.5294013892421578);
    assert_eq!(error, 0.14432099937166218);

    ai.training_step(data.iter());
    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 1, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.5369993099938283);
    assert_eq!(error, 0.1386058877965444);

    ai.training_step(data.iter());
    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 2, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.5444452723304654);
    assert_eq!(error, 0.13311709463980123);

    ai.training_step(data.iter());
    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 3, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.5517423154203697);
    assert_eq!(error, 0.1278456576920651);

    ai.training_step(data.iter());
    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 4, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.5588934176484759);
    assert_eq!(error, 0.12278296964745934);

    ai.training_step(data.iter());
    let TestsResult { outputs, error, .. } = ai.test_propagate(data.iter());
    println!("epoch: 5, output: {}, error: {}", outputs[0].0[0], error);
    assert_eq!(outputs[0].0[0], 0.56590149783202);
    assert_eq!(error, 0.11792076404941992);

    println!("ai: {}", ai);

    /*
    let res_name = format!("gen{}_result", epoch);
    test_res.export_to_js(&mut js_file, &res_name);
    result_names.push(&res_name).export(&mut js_file);
    */
}
