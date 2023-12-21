use rusty_ai::{
    bias::LayerBias,
    data::PairList,
    loss_function::{LossFunction, SquaredError},
    matrix::Matrix,
    neural_network::NNBuilder,
    optimizer::sgd::SGD,
    trainer::NNTrainer,
    ActivationFn, BuildLayer, Input, Norm, Optimizer,
};

struct Args<'a, F, L, O>
where
    F: Fn(usize) -> bool,
    L: LossFunction<1, ExpectedOutput = [f64; 1]>,
    O: Optimizer,
{
    ai: NNTrainer<1, 1, L, O>,
    data: PairList<1, [f64; 1]>,
    losses: &'a [f64],
    epochs: usize,
    test_condition: F,
}

fn test<F, L, O>(args: Args<F, L, O>)
where
    F: Fn(usize) -> bool,
    L: LossFunction<1, ExpectedOutput = [f64; 1]>,
    O: Optimizer,
{
    let Args { mut ai, data, losses, epochs, test_condition } = args;
    let test = |epoch: usize, ai: &NNTrainer<1, 1, L, O>, expected_loss: f64| {
        let error = ai.test_batch(data.iter()).map(|(_, loss)| loss).sum::<f64>();

        println!("epoch: {:>4}, loss: {:<20} {:064b}", epoch, error, error.to_bits());
        println!("    expected_loss: {:<20} {:064b}", expected_loss, expected_loss.to_bits());
        println!(
            "{}diff:
        {:064b}\n",
            " ".repeat(34),
            error.to_bits() ^ expected_loss.to_bits()
        );
        let diff = (error - expected_loss).abs();
        println!("diff: {} = 10^{}", diff, diff.log10());
        assert!(diff < 1e-14);
    };

    let mut losses = losses.into_iter().copied();

    test(0, &ai, losses.next().unwrap());

    for epoch in 1..epochs + 1 {
        ai.train(&data).execute();
        if test_condition(epoch) {
            test(epoch, &ai, losses.next().unwrap());
        }
    }
}

#[test]
fn sine() {
    fn vec<const W: usize, const H: usize>(arr: [[f64; W]; H]) -> Vec<Vec<f64>> {
        arr.into_iter().map(Vec::from).collect()
    }

    let w1 = Matrix::from_rows(vec([
        [0.4365753188883896],
        [0.9413543661345566],
        [-0.3656366713056675],
        [0.42030797989253815],
        [0.2854215078862145],
    ]));
    let b1 = LayerBias::from(
        [
            -0.2996289431359356,
            0.986120857506954,
            0.45516228678556825,
            0.8055778757470997,
            -0.689708620272808,
        ]
        .to_vec(),
    );
    let w2 = Matrix::from_rows(vec([
        [
            0.21467640376231029,
            0.05910216116729297,
            -0.07270556287403136,
            0.09825272492033317,
            0.10346141570152735,
        ],
        [
            0.2291406279769201,
            0.32016330811210963,
            0.3251660189616064,
            0.03309772335563605,
            0.13220300987004602,
        ],
        [
            0.16394744035632014,
            -0.3340153518421561,
            -0.22797579801153234,
            0.31974864353684423,
            -0.33682344392514063,
        ],
        [
            -0.430209621183488,
            -0.074278323717619,
            0.3940483082076613,
            0.44169077835110265,
            -0.37663595792134724,
        ],
        [
            0.26064684922258663,
            -0.16759093540588033,
            0.029746026275235732,
            -0.4079191271614241,
            0.18522885908525116,
        ],
    ]));
    let b2 = LayerBias::from(
        [
            -0.09097069414373553,
            -0.37051233024050073,
            0.4075522107849301,
            0.30303153207404837,
            0.383553520871672,
        ]
        .to_vec(),
    );
    let w3 = Matrix::from_rows(vec([
        [
            -0.08604004294389384,
            -0.2111917324479703,
            -0.17275450246045412,
            -0.10881427108274791,
            0.4463921558265414,
        ],
        [
            -0.08439520484841244,
            0.009691723555064697,
            -0.24242316756653387,
            0.02187182642556704,
            -0.37791587336957994,
        ],
        [
            0.34344419294447126,
            -0.009047235469132055,
            -0.4272871083737448,
            0.16802974723329936,
            0.33169105615064515,
        ],
        [
            0.19713276051735834,
            -0.06764415687471487,
            0.14139433958202735,
            -0.11688751837120669,
            -0.28736563948463684,
        ],
        [
            0.13912682303335583,
            0.41568545404825996,
            0.13951559923546403,
            0.3271228095344742,
            -0.09826785888723547,
        ],
    ]));
    let b3 = LayerBias::from(
        [
            -0.12373051458940326,
            -0.2744419905274138,
            0.33420237184888146,
            -0.17581290357712187,
            -0.3956511222568767,
        ]
        .to_vec(),
    );
    let w4 = Matrix::from_rows(vec([
        [
            0.4146347638085144,
            0.38916855376235837,
            0.2516861334693849,
            0.3809065786052524,
            0.052020387870368424,
        ],
        [
            -0.30480522545422184,
            0.33972006405452637,
            0.2361691216991906,
            -0.048498669522668035,
            0.26393427116689283,
        ],
        [
            -0.06685733802004208,
            -0.29678636278772846,
            -0.1676431954195087,
            -0.2249926955632583,
            -0.3037004232040834,
        ],
        [
            -0.17590703171604363,
            -0.255058790662183,
            -0.3897411540974514,
            0.007808289696270137,
            -0.1287817222821281,
        ],
        [
            0.40222205746772954,
            -0.1833606122825174,
            -0.29848245341537594,
            0.38662907105916866,
            0.08819483857664974,
        ],
    ]));
    let b4 = LayerBias::from(
        [
            0.4170405541187824,
            0.05754973781938314,
            -0.19269313179873393,
            0.10500156759133414,
            0.12977469128854874,
        ]
        .to_vec(),
    );
    let w5 = Matrix::from_rows(vec([[
        0.3413003170848592,
        0.12496182793345802,
        0.15006687729362725,
        -0.1508118644190267,
        -0.1315805547665502,
    ]]));
    let b5 = LayerBias::from([-0.1936956647543538].to_vec());
    let x = [
        [-2.0],
        [-1.5555555555555556],
        [-1.1111111111111112],
        [-0.6666666666666667],
        [-0.22222222222222232],
        [0.22222222222222232],
        [0.6666666666666667],
        [1.1111111111111112],
        [1.5555555555555556],
        [2.0],
    ];
    let y = [
        [-0.9092974268256817],
        [-0.9998838616941024],
        [-0.8961922010299563],
        [-0.6183698030697371],
        [-0.22039774345612237],
        [0.22039774345612237],
        [0.6183698030697371],
        [0.8961922010299563],
        [0.9998838616941024],
        [0.9092974268256817],
    ];
    let epochs = 20;
    let test_spread = 1;
    let losses = [
        6.091501566998738,
        6.059195427291992,
        5.994109625076152,
        5.899444459222711,
        5.7918522031969575,
        5.668584175895945,
        5.5288589940429365,
        5.371752448962822,
        5.19321843952137,
        4.992371634371672,
        4.771541137402648,
        4.531298660095017,
        4.269436620813997,
        3.987220386588473,
        3.689111330690796,
        3.378347636906795,
        3.057692062480944,
        2.73236603734566,
        2.407151901315005,
        2.088005411275766,
        1.7792481476766535,
    ];

    let ai = NNBuilder::default()
        .default_activation_function(ActivationFn::ReLU)
        .input::<1>()
        .layer_from_parameters(w1, b1)
        .layer_from_parameters(w2, b2)
        .layer_from_parameters(w3, b3)
        .layer_from_parameters(w4, b4)
        .layer_from_parameters(w5, b5)
        .identity()
        .build()
        .to_trainer()
        .loss_function(SquaredError)
        .optimizer(SGD { learning_rate: 0.01, ..SGD::default() })
        .retain_gradient(true)
        .new_clip_gradient_norm(5.0, Norm::Two)
        .build();

    let x = x.into_iter().map(Input::from);
    let data = PairList::new(x, y);

    println!("{:?}", data);

    test(Args {
        ai,
        data,
        losses: &losses,
        epochs,
        test_condition: |epoch| epoch % test_spread == 0,
    });
}