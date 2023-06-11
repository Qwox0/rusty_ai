use rusty_ai::prelude::*;

fn test_ai<const IN: usize, const OUT: usize>(
    ai: &TrainableNeuralNetwork<IN, OUT>,
    data: &PairList<IN, OUT>,
    epoch: usize,
) -> TestsResult<OUT> {
    let res = ai.test_propagate(data.iter());
    println!(
        "epoch: {}, output: {}, error: {}",
        epoch, res.outputs[0].0[0], res.error
    );
    return res;
}

#[test]
fn xor() {
    fn vec<const W: usize, const H: usize>(arr: [[f64; W]; H]) -> Vec<Vec<f64>> {
        arr.into_iter().map(Vec::from).collect()
    }

    let w1 = Matrix::from_rows(vec([
        [0.3087053684846597, 0.6656380557933089],
        [-0.2585441697307142, 0.29720262276883275],
        [0.20182348372283193, -0.21186965753117842],
    ]));
    let b1 = LayerBias::OnePerNeuron(
        [0.6972927454126603, 0.32184833952645137, 0.5696295787146282].to_vec(),
    );
    let w2 = Matrix::from_rows(vec([
        [-0.39820345757691106, 0.277146045531709, 0.07630056197526414],
        [
            -0.09386247806258657,
            0.1268437224453163,
            0.13356811332901933,
        ],
        [0.2958192786980403, 0.4133290534616249, 0.4197875253969556],
    ]));
    let b2 = LayerBias::OnePerNeuron(
        [
            0.04272897711791958,
            0.17067335184839438,
            0.21165523538445397,
        ]
        .to_vec(),
    );
    let w3 = Matrix::from_rows(vec([
        [-0.4312119650207605, -0.2943154896790036, 0.4127937237968699],
        [
            -0.4348371963114325,
            -0.5553982327406168,
            -0.09589290358084727,
        ],
        [0.5087141784298266, 0.5702203429090708, -0.48623493087075187],
    ]));
    let b3 = LayerBias::OnePerNeuron(
        [
            0.33649363542683797,
            -0.21635896726743253,
            0.03840195479327876,
        ]
        .to_vec(),
    );
    let w4 = Matrix::from_rows(vec([[
        -0.5266213286985537,
        0.23912942882472604,
        -0.11744266113720543,
    ]]));
    let b4 = LayerBias::OnePerNeuron([-0.47832936152865413].to_vec());

    let relu = ActivationFn::default_relu();

    let mut ai = NeuralNetworkBuilder::default()
        .input_layer::<2>()
        .hidden_layer(LayerBuilder::with_weights(w1).bias(b1))
        .hidden_layer(LayerBuilder::with_weights(w2).bias(b2))
        .hidden_layer(LayerBuilder::with_weights(w3).bias(b3))
        .output_layer(
            LayerBuilder::with_weights(w4)
                .bias(b4)
                .activation_function(ActivationFn::Identity)
        )
        .error_function(ErrorFunction::SquaredError)
        .sgd_optimizer(GradientDescent {
            learning_rate: 0.01,
        })
        .build();

    println!("ai: {}", ai);

    //let data = DataList::random_simple(2000, -PI..PI, f64::sin);
    let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = [[0.0], [1.0], [1.0], [0.0]];
    let data: Vec<_> = x.into_iter().zip(y).map(Pair::from).collect();
    let data = PairList::from(data);
    println!("\ndata: {:?}\n", data);

    let res = ai.test_propagate(data.iter());
    println!(
        "epoch: {}, output: {:?}, error: {}",
        0, res.outputs, res.error
    );
    assert_eq!(res.error, 8.103664081622888);
    let res: Vec<_> = res.outputs.into_iter().map(|a| a.0[0]).collect();
    assert_eq!(res[0], -0.7928575734943181);
    assert_eq!(res[1], -0.8512594988704734);
    assert_eq!(res[2], -0.8132921938214386);
    assert_eq!(res[3], -0.871694119197594);

    println!("END");
    println!("cpu: {}", rusty_ai::util::cpu_count());

    assert!(false);
    for epoch in 1..=1000 {
        ai.training_step(data.iter());

        if epoch % 100 == 0 {
            test_ai(&ai, &data, epoch);
        }
    }
    println!("ai: {}", ai);
}
