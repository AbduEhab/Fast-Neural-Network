use criterion::{criterion_group, criterion_main, Criterion};

use fast_neural_network::{activation::*, neural_network::*};

use ndarray::*;

const MATRIX_SIZE: usize = 1000;

fn ndarray_dot(c: &mut Criterion) {
    let mut matrix_a = Array2::<f64>::zeros((MATRIX_SIZE, MATRIX_SIZE));
    let mut matrix_b = Array2::<f64>::zeros((MATRIX_SIZE, MATRIX_SIZE));

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a[[i, j]] = 1. + i as f64 + j as f64;
            matrix_b[[i, j]] = 1. + i as f64 + j as f64;
        }
    }

    c.bench_function("ndarray dot", |b| b.iter(|| matrix_a.dot(&matrix_b)));
}

fn simple_feedforward(c: &mut Criterion) {
    let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);

    network.add_hidden_layer_with_size(4);
    network.add_hidden_layer_with_size(4);
    network.compile();

    let input = array!(2., 1., -1.);

    c.bench_function("simple feedforward", |b| b.iter(|| network.forward(&input)));
}

fn simple_feedbackward(c: &mut Criterion) {
    let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);

    network.add_hidden_layer_with_size(4);
    network.add_hidden_layer_with_size(4);
    network.compile();

    let layer_1_weights = Array::from_shape_vec(
        (4, 3),
        vec![
            0.03, 0.62, 0.85, 0.60, 0.62, 0.64, 0.75, 0.73, 0.34, 0.46, 0.14, 0.06,
        ],
    )
    .unwrap();
    let layer_1_biases = array!(0.14, 0.90, 0.65, 0.32);
    let layer_2_weights = Array::from_shape_vec(
        (4, 4),
        vec![
            0.90, 0.95, 0.26, 0.70, 0.12, 0.84, 0.58, 0.78, 0.92, 0.16, 0.49, 0.90, 0.64, 0.60,
            0.64, 0.85,
        ],
    )
    .unwrap();
    let layer_2_biases = array!(0.41, 0.09, 0.28, 0.70);
    let layer_3_weights = Array::from_shape_vec((1, 4), vec![0.23, 0.34, 0.24, 0.67]).unwrap();
    let layer_3_biases = array!(0.23);

    network.set_layer_weights(0, layer_1_weights);
    network.set_layer_biases(0, layer_1_biases);
    network.set_layer_weights(1, layer_2_weights);
    network.set_layer_biases(1, layer_2_biases);
    network.set_layer_weights(2, layer_3_weights);
    network.set_layer_biases(2, layer_3_biases);

    // let input: ndarray::Array1<f64> = array!(2., 1., -1.);

    // let prediction = network.forward(&input);

    c.bench_function("simple feedbackward", |b| b.iter(|| network.train(&[(array!(2., 1., -1.), array!(9.))], 1, usize::MAX)));
    //     network.train(&[(input.clone(), array!(9.))], 1, usize::MAX);

    //     let new_prediction = network.forward(&input);

    //     println!("{:?}", prediction);
    //     println!("{:?}", new_prediction);

    //     println!("{}", network);

    //     network.save("network.json");

    //     let mut network = Network::load("network.json");

    //     println!("{:?}", network.forward(&input));
}

criterion_group!(benches, ndarray_dot, simple_feedforward, simple_feedbackward);
criterion_main!(benches);
