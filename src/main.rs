use fast_neural_network::{activation::*, neural_network::*};
use ndarray::*;

fn main() {
    let mut network = Network::new(2, 1, ActivationType::Tanh, 0.01); // Create a new network with 2 inputs, 1 output, a LeakyRelu activation function, and a learning rate of 0.01

    network.add_hidden_layer_with_size(3); // Add a hidden layer with 2 neurons

    network.compile();  // Compile the network to prepare it for training
                        // (will be done automatically during training)
                        // The API is exposed so that the user can compile
                        // the network on a different thread before training if they want to

    // Let's create a dataset to represent the XOR function
    let mut dataset: Vec<(ndarray::Array1<f64>, ndarray::Array1<f64>)> = Vec::new();

    dataset.push((array!(0., 0.), array!(1.)));
    dataset.push((array!(1., 0.), array!(0.)));
    dataset.push((array!(0., 1.), array!(0.)));
    dataset.push((array!(1., 1.), array!(1.)));

    network.train(&dataset, 20_000, 1_000); // train the network for 20,000 epochs with a decay_time of 1,000 epochs

    let mut res;

    // Let's check the result
    for i in 0..dataset.len() {
        res = network.forward(&dataset[i].0);
        let d = &dataset[i];
        println!(
            "for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
            d.0[0], d.0[1], d.1[0], res
        );
    }

    network.save("network.json"); // Save the model as a json to a file

    // Load the model from a json file using the below line
    // let mut loaded_network = Network::load("network.json");
}

// fn main() {
//     let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);

//     network.add_hidden_layer_with_size(4);
//     network.add_hidden_layer_with_size(4);
//     network.compile();

//     let layer_1_weights = Array::from_shape_vec(
//         (4, 3),
//         vec![
//             0.03, 0.62, 0.85, 0.60, 0.62, 0.64, 0.75, 0.73, 0.34, 0.46, 0.14, 0.06,
//         ],
//     )
//     .unwrap();
//     let layer_1_biases = array!(0.14, 0.90, 0.65, 0.32);
//     let layer_2_weights = Array::from_shape_vec(
//         (4, 4),
//         vec![
//             0.90, 0.95, 0.26, 0.70, 0.12, 0.84, 0.58, 0.78, 0.92, 0.16, 0.49, 0.90, 0.64, 0.60,
//             0.64, 0.85,
//         ],
//     )
//     .unwrap();
//     let layer_2_biases = array!(0.41, 0.09, 0.28, 0.70);
//     let layer_3_weights = Array::from_shape_vec((1, 4), vec![0.23, 0.34, 0.24, 0.67]).unwrap();
//     let layer_3_biases = array!(0.23);

//     network.set_layer_weights(0, layer_1_weights);
//     network.set_layer_biases(0, layer_1_biases);
//     network.set_layer_weights(1, layer_2_weights);
//     network.set_layer_biases(1, layer_2_biases);
//     network.set_layer_weights(2, layer_3_weights);
//     network.set_layer_biases(2, layer_3_biases);

//     let input: ndarray::Array1<f64> = array!(2., 1., -1.);

//     let prediction = network.forward(&input);

//     network.train(&[(input.clone(), array!(9.))], 1, usize::MAX);

//     let new_prediction = network.forward(&input);

//     println!("{:?}", prediction);
//     println!("{:?}", new_prediction);

//     println!("{}", network);

//     network.save("network.json");

//     let mut network = Network::load("network.json");

//     println!("{:?}", network.forward(&input));
// }
