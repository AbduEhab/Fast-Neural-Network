# Rust-Neural-Network

This is the beginning of a neural network library written in Rust, designed to provide a flexible and efficient platform for building and training neural networks.
The current implementation requires that everything is allocated on the heap and be computed partially. This will be changed once Rust has a better implementation of Generics in Constant Expressions.

> The nightly compiler is not something I wanna bother with.

## Project Status

> Current Version: 0.2.2 (beta)

This library is still in its early development stages, and the current version is in the beta stage and will jump to a 1.0.0 version once stack-based allocations are implemented.
Contributions and feedback are welcome, but please be aware that the internal structure may undergo significant changes as the library matures, so don't depend on the internal `Matrix` implementation as it will most likely change.

## Features

- [x] Basic Neural Network Layers: The library currently supports fundamental neural network layers such as fully connected (dense) layers and convolutional layers (I call them hidden layers in the API)
- [x] Activation functions (Sigmoid, Tanh, ArcTanh, Relu, LeakyRelu, SoftMax, SoftPlus).
- [x] Training: The library allows for low-level training of the network using backpropagation and gradient descent.
- [x] Model Serialization

But, here's the example of creating a simple neural network and then training it for a single epoch using the library

```rust
use fast_neural_network::{activation::*, matrix::*, neural_network::*};

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

```

## Speed

The focus of this library is multi-threaded performance. The library is designed to be as fast as possible, and I have done my best to optimize the code for performance. The library is still in its early stages, so there is still room for improvement, but I have done my best to make it as fast as possible. I just wish Rust had a better implementation of Generics in Constant Expressions like C++.

> Matrix parallelization is currently not implemented, but it will be once better generics are implemented in Rust.

## Contributing

Contributions are highly encouraged! If you're interested in adding new features, improving performance, fixing bugs, or enhancing documentation, I would appreciate your help. Just open a pull request and I'll look into it.

## Roadmap

The following features and *might* be implemented in a future releases:

- Support for more activation functions
- GPU acceleration using CUDA or similar technologies (probably just shaders but idk it seems hard)
- Enhanced model evaluation tools (and possibly, maybe a GUI to go with them. If I write one it will be in raylib btw)
