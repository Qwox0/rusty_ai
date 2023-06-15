use std::ops::Range;

use rusty_ai::{data::DataBuilder, prelude::*};

#[test]
fn sine() {
    fn vec<const W: usize, const H: usize>(arr: [[f64; W]; H]) -> Vec<Vec<f64>> {
        arr.into_iter().map(Vec::from).collect()
    }

    let w1 = Matrix::from_rows(vec([
        [0.4365753188883896],
        [0.9413543661345566],
        [-0.3656366713056675],
    ]));
    let b1 = LayerBias::OnePerNeuron(
        [0.4203079798925382, 0.28542150788621456, -0.2996289431359356].to_vec(),
    );
    let w2 = Matrix::from_rows(vec([
        [0.5693371425351444, 0.26278806880061345, 0.4651006034157949],
        [-0.39820345757691106, 0.277146045531709, 0.07630056197526414],
        [
            -0.09386247806258657,
            0.1268437224453163,
            0.13356811332901933,
        ],
    ]));
    let b2 = LayerBias::OnePerNeuron(
        [0.29581927869804037, 0.413329053461625, 0.4197875253969557].to_vec(),
    );
    let w3 = Matrix::from_rows(vec([
        [
            0.042728977117919564,
            0.17067335184839436,
            0.2116552353844539,
        ],
        [-0.4312119650207605, -0.2943154896790036, 0.4127937237968699],
        [
            -0.4348371963114325,
            -0.5553982327406168,
            -0.09589290358084727,
        ],
    ]));
    let b3 = LayerBias::OnePerNeuron(
        [0.5087141784298267, 0.5702203429090709, -0.4862349308707519].to_vec(),
    );
    let w4 = Matrix::from_rows(vec([[
        0.3364936354268379,
        -0.2163589672674325,
        0.03840195479327875,
    ]]));
    let b4 = LayerBias::OnePerNeuron([-0.5266213286985538].to_vec());
    let loss0 = 66.24718244759788;
    let loss1 = 61.23976940360403;

    let mut ai = NeuralNetworkBuilder::default()
        .default_activation_function(ActivationFn::ReLU(0.0))
        .input::<1>()
        .layer_with_weights_and_bias(w1, b1)
        .layer_with_weights_and_bias(w2, b2)
        .layer_with_weights_and_bias(w3, b3)
        .default_activation_function(ActivationFn::Identity)
        .layer_with_weights_and_bias(w4, b4)
        .output()
        .error_function(ErrorFunction::SquaredError)
        .sgd_optimizer(GradientDescent {
            learning_rate: 0.01,
        })
        .clip_gradient_norm(5.0, Norm::Two)
        .build();

    println!("ai: {}", ai);

    fn linspace(range: Range<f64>, data_count: usize) -> Vec<[f64; 1]> {
        let data_with = range.end - range.start;
        let data_diff = data_with / (data_count - 1) as f64;

        (0..data_count)
            .map(|x| x as f64 * data_diff + range.start)
            .map(|x| [x])
            .collect()
    }

    let data_count = 100;
    let x = linspace(-2.0..2.0, data_count);
    println!("{x:?}");
    let data = PairList::with_fn(x, |x| [x[0].sin()]);
    println!("{data:?}");

    let res0 = ai.test_propagate(data.iter());
    println!(
        "{:?}",
        res0.outputs.iter().map(|x| x.0[0]).collect::<Vec<_>>()
    );
    println!("epoch: 0, loss: {}", res0.error);
    assert!(res0.error - loss0 < 10e-13);

    //ai.training_step(&data);

    let res1 = ai.test_propagate(data.iter());
    println!("epoch: 1, loss: {}", res1.error);
    assert!(res1.error - loss1 < 10e-13);
}
