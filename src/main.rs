use fast_neural_network::{activation::*, neural_network::*};
use ndarray::*;

fn main() {
    let mut network = Network::new(2, 1, ActivationType::Tanh, 0.005); // Create a new network with 2 inputs, 1 output, a tanh activation function, and a learning rate of 0.005

    network.add_hidden_layer_with_size(2); // Add a hidden layer with 2 neurons

    network.compile();  // Compile the network to prepare it for training
                        // (will be done automatically during training)
                        // The API is exposed so that the user can compile
                        // the network on a different thread before training if they want to

    // Let's create a dataset to represent the XOR function
    let mut dataset: Vec<(ndarray::Array1<f64>, ndarray::Array1<f64>)> = Vec::new();

    dataset.push((array!(0., 0.), array!(0.)));
    dataset.push((array!(1., 0.), array!(1.)));
    dataset.push((array!(0., 1.), array!(1.)));
    dataset.push((array!(1., 1.), array!(0.)));

    network.train(&dataset, 50_000, 5_000); // train the network for 50,000 epochs with a decay_time of 5,000 epochs

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