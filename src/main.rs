use fast_neural_network::{activation::*, neural_network::*};
use ndarray::*;

fn main() {
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
    let layer_1_biases = Array::<f64, _>::from_shape_vec(4, vec![0.14, 0.90, 0.65, 0.32]).unwrap();
    let layer_2_weights = Array::from_shape_vec(
        (4, 4),
        vec![
            0.90, 0.95, 0.26, 0.70, 0.12, 0.84, 0.58, 0.78, 0.92, 0.16, 0.49, 0.90, 0.64, 0.60,
            0.64, 0.85,
        ],
    )
    .unwrap();
    let layer_2_biases = Array::from_shape_vec(4, vec![0.41, 0.09, 0.28, 0.70]).unwrap();
    let layer_3_weights = Array::from_shape_vec((1, 4), vec![0.23, 0.34, 0.24, 0.67]).unwrap();
    let layer_3_biases = Array::from_shape_vec(1, vec![0.23]).unwrap();

    network.set_layer_weights(0, layer_1_weights);
    network.set_layer_biases(0, layer_1_biases);
    network.set_layer_weights(1, layer_2_weights);
    network.set_layer_biases(1, layer_2_biases);
    network.set_layer_weights(2, layer_3_weights);
    network.set_layer_biases(2, layer_3_biases);

    let input: ndarray::Array1<f64> = Array::from_vec(vec![2., 1., -1.]);

    let prediction = network.forward_propagate(&input);

    network.back_propagate(&input, &Array::from_vec(vec![9.0]));

    let new_prediction = network.forward_propagate(&input);

    println!("{:?}", prediction);
    println!("{:?}", new_prediction);

    println!("{}", network);

    network.save("network.json");

    let mut network = Network::load("network.json");

    println!("{:?}", network.forward_propagate(&input));
}
