use fast_neural_network::neural_network::*;
use fast_neural_network::activation::*;
use fast_neural_network::matrix::*;

fn main() {
    let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);

    network.add_hidden_layer_with_size(4);
    network.add_hidden_layer_with_size(4);
    network.compile();

    let layer_1_weights = Matrix::from_vec(
        vec![
            0.03, 0.62, 0.85, 0.60, 0.62, 0.64, 0.75, 0.73, 0.34, 0.46, 0.14, 0.06,
        ],
        4,
        3,
    );
    let layer_1_biases = Matrix::from_vec(vec![0.14, 0.90, 0.65, 0.32], 4, 1);
    let layer_2_weights = Matrix::from_vec(
        vec![
            0.90, 0.95, 0.26, 0.70, 0.12, 0.84, 0.58, 0.78, 0.92, 0.16, 0.49, 0.90, 0.64, 0.60,
            0.64, 0.85,
        ],
        4,
        4,
    );
    let layer_2_biases = Matrix::from_vec(vec![0.41, 0.09, 0.28, 0.70], 4, 1);
    let layer_3_weights = Matrix::from_vec(vec![0.23, 0.34, 0.24, 0.67], 1, 4);
    let layer_3_biases = Matrix::from_vec(vec![0.23], 1, 1);

    network.set_layer_weights(0, layer_1_weights);
    network.set_layer_biases(0, layer_1_biases);
    network.set_layer_weights(1, layer_2_weights);
    network.set_layer_biases(1, layer_2_biases);
    network.set_layer_weights(2, layer_3_weights);
    network.set_layer_biases(2, layer_3_biases);

    let input: Vec<f64> = vec![2., 1., -1.];

    let prediction = network.forward_propagate(&input);
    network.back_propagate(&input, &vec![9.0]);
    let new_prediction = network.forward_propagate(&input);

    println!("{:?}", prediction);
    println!("{:?}", new_prediction);

    network.save("network.json");

    let mut network = Network::load("network.json");

    println!("{:?}", network.forward_propagate(&input));
}
