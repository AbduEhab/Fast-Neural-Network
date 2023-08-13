# Rust-Neural-Network

This is the beginning of a neural network library written in Rust, designed to provide a flexible and efficient platform for building and training neural networks. 
The current implementation requires that everything is allocated on the heap and be computed partially. This will be changed once Rust has a better implementation of Generics in Constant Expressions.

> The nightly compiler is not something I wanna bother with.

Project Status
Current Version: 0.1.0 (beta)

This library is still in its early development stages, and the current version is in the beta stage and will jump to a 1.0 version once stack-based allocations are implemented. 
Contributions and feedback are welcome, but please be aware that the internal structure may undergo significant changes as the library matures, so don't depend on the internal `Matrix` implementation as it will most likely change.

Features
Basic Neural Network Layers: The library currently supports fundamental neural network layers such as fully connected (dense) layers, convolutional layers (I call them hidden layers in the API), and activation functions (ReLU, sigmoid, etc.).

Backpropagation: The library includes an implementation of backpropagation, which is crucial for training neural networks. This allows the network to learn from data and update its weights accordingly.

Model Serialization: I plan to support model serialization to allow users to save and load trained models easily. I have started testing `sendre` in the meantime.

Documentation: I'll write the documentation once I'm ready to publish to crates.io. For now, the example left in the `main.rs` file should be more than enough.

But, here's the example of creating a simple feedforward neural network using the library, just for those who don't have the time to browse the file.:

```rust
    let mut network = Network::empty_network(3, 1, ActivationType::Relu, 0.005);

    network.add_hidden_layer_with_size(4);
    network.add_hidden_layer_with_size(4);
    network.compile();

    let layer_1_weights = Matrix::from_vec(
        vec![
            0.03, 0.62, 0.85,
            0.60, 0.62, 0.64,
            0.75, 0.73, 0.34,
            0.46, 0.14, 0.06,
        ],
        4,
        3,
    );
    let layer_1_biases = Matrix::from_vec(vec![0.14, 0.90, 0.65, 0.32], 4, 1);
    let layer_2_weights = Matrix::from_vec(
        vec![
            0.90, 0.95, 0.26, 0.70,
            0.12, 0.84, 0.58, 0.78,
            0.92, 0.16, 0.49, 0.90,
            0.64, 0.60, 0.64, 0.85,
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
    let error = network.back_propagate(&input, &vec![9.0]);
    let new_prediction = network.forward_propagate(&input);

    println!("{}", network);
    println!("{:?}", prediction);
    println!("{:?}", error);
    println!("{:?}", new_prediction);
}
```

Contributing
Contributions are highly encouraged! If you're interested in adding new features, improving performance, fixing bugs, or enhancing documentation, I would appreciate your help. Just open a pull request and I'll look into it.

Roadmap
The following features and improvements are planned for future releases:

Support for more activation functions
GPU acceleration using CUDA or similar technologies (probably just shaders but idk it seems hard)
Enhanced model evaluation tools (and possibly, maybe a GUI to go with them. If I write one it will be in raylib btw)
