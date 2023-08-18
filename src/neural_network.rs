//! # Neural Network
//!
//! This is the neural network module. It contains the `Network` struct and the `ActivationType` enum. This is the heart of the crate.
//!

use rand::random;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::f64;
use std::fmt::{self, Display, Formatter};

use crate::activation::*;
use crate::matrix::*;

/// a trait for adding two vectors
pub trait LinearAlgebra {
    fn add(&self, other: &Self) -> Self;
}

impl LinearAlgebra for Vec<f64> {
    fn add(&self, other: &Self) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = vec![0.0; self.len()];
        for i in 0..self.len() {
            result[i] = self[i] + other[i];
        }
        result
    }
}

/// The underlying layer struct for the network.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Layer {
    pub size: usize,
    pub bias: Vec<f64>,
}

impl Layer {
    /// Creates a new layer with the given size.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::neural_network::*;
    ///
    /// let layer = Layer::new(3);
    ///
    /// assert_eq!(layer.size, 3);
    /// assert_eq!(layer.bias.len(), 3);
    /// ```
    pub fn new(size: usize) -> Self {
        Layer {
            size,
            bias: (0..size).into_par_iter().map(|_| random::<f64>()).collect(),
        }
    }

    /// Creates a new layer with the given bias.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::neural_network::*;
    ///
    /// let layer = Layer::from_vec(vec![0.03, 0.62, 0.85, 0.60, 0.62, 0.64]);
    ///
    /// assert_eq!(layer.size, 6);
    /// assert_eq!(layer.bias.len(), 6);
    /// ```
    pub fn from_vec(vec: Vec<f64>) -> Self {
        Layer {
            size: vec.len(),
            bias: vec,
        }
    }
}

/// The main neural network struct.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Network {
    inputs: Layer,                         // number of neurons in input layer
    outputs: Layer,                        // number of neurons in output layer
    hidden_layers: Vec<Layer>, // number of hidden layers (each layer has a number of neurons)
    layer_matrices: Vec<(Matrix, Matrix)>, // (weights, biases)
    activation_matrices: Vec<Matrix>,
    activation: ActivationType, // activation function
    leanring_rate: f64,
    compiled: bool,
}

impl Network {
    /// Creates a new empty network with the given number of inputs and outputs.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};

    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    ///
    /// assert_eq!(network.dimensions().0, 3);
    /// assert_eq!(network.dimensions().1, 1);
    /// assert_eq!(network.hidden_layers_size(), 0);
    /// assert_eq!(network.leanring_rate(), 0.005);
    /// ```
    pub fn new(inputs: usize, outputs: usize, activation_func: ActivationType, alpha: f64) -> Self {
        Network {
            inputs: Layer::new(inputs),
            outputs: Layer::new(outputs),
            hidden_layers: vec![],
            layer_matrices: vec![],
            activation_matrices: vec![],
            activation: activation_func,
            leanring_rate: alpha,
            compiled: false,
        }
    }

    /// Creates a new empty network with the given number of inputs and outputs and the given hidden layers.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};
    ///
    /// let mut network = Network::new_with_layers(3, 1, vec![Layer::new(4), Layer::new(4)], ActivationType::Relu, 0.005);
    ///
    /// assert_eq!(network.dimensions().0, 3);
    /// assert_eq!(network.dimensions().1, 1);
    /// assert_eq!(network.hidden_layers_size(), 2);
    /// assert_eq!(network.leanring_rate(), 0.005);
    /// ```
    pub fn new_with_layers(
        inputs: usize,
        outputs: usize,
        hidden_layers: Vec<Layer>,
        activation_func: ActivationType,
        alpha: f64,
    ) -> Self {
        Network {
            inputs: Layer::new(inputs),
            outputs: Layer::new(outputs),
            hidden_layers,
            layer_matrices: vec![],
            activation_matrices: vec![],
            activation: activation_func,
            leanring_rate: alpha,
            compiled: false,
        }
    }

    /// Loads a network from a json file.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::neural_network::*;
    ///
    /// let mut network = Network::load("network.json");
    /// ```
    pub fn load(path: &str) -> Self {
        let json = std::fs::read_to_string(path).expect("Unable to read file");
        serde_json::from_str(&json).unwrap()
    }

    /// Saves the network to a json file.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};
    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    ///
    /// network.save("network.json");
    /// ```
    pub fn save(&self, path: &str) {
        let json = serde_json::to_string(&self).unwrap();
        std::fs::write(path, json).expect("Unable to write file");
    }

    /// Creates a network from the given JSON string.
    ///
    /// ## Panics
    ///
    /// Panics if the JSON string is not valid.
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(&json).unwrap()
    }

    /// Adds a hidden layer to the network.
    pub fn add_hidden_layer(&mut self, layer: Layer) {
        self.hidden_layers.push(layer);
    }

    /// Adds a hidden layer to the network with the given size.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};
    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    ///
    /// network.add_hidden_layer_with_size(4);
    /// ```
    pub fn add_hidden_layer_with_size(&mut self, size: usize) {
        self.hidden_layers.push(Layer::new(size));
    }

    /// Compiles the network. This is done automatically during training.
    /// > Compilations should be done after the hidden layers are set.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};
    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    /// network.add_hidden_layer_with_size(4);
    /// network.add_hidden_layer_with_size(4);
    /// network.compile();
    /// ```
    ///
    /// ## Panics
    ///
    /// Panics if any of the dimentions is 0
    pub fn compile(&mut self) {
        if cfg!(debug_assertions) {
            debug_assert!(self.inputs.size > 0);
            debug_assert!(self.outputs.size > 0);
        }

        if self.compiled {
            return;
        }

        self.layer_matrices.clear();

        let mut layers = vec![self.inputs.clone()];
        layers.append(&mut self.hidden_layers.clone());
        layers.push(self.outputs.clone());

        for i in 0..layers.len() - 1 {
            let mut weights = vec![];
            let mut biases = vec![];
            for _ in 0..layers[i + 1].size {
                weights.append(
                    &mut (0..layers[i].size)
                        .into_par_iter()
                        .map(|_| random::<f64>())
                        .collect(),
                );
                biases.push(random::<f64>());
            }
            let weights = Matrix::from_vec(weights, layers[i + 1].size, layers[i].size);
            let biases = Matrix::from_vec(biases, layers[i + 1].size, 1);
            self.layer_matrices.push((weights, biases));
        }

        self.compiled = true;
    }

    /// Returns a Tuple with the dimentions of the Neural Network (inputs, outputs)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.inputs.size, self.outputs.size)
    }

    /// Sets the activation function to be used by the network
    pub fn set_activation(&mut self, activation: ActivationType) {
        self.activation = activation;
    }

    /// Returns the activation function being used
    pub fn activation(&self) -> ActivationType {
        self.activation.clone()
    }

    /// Sets the weights and biases of the given layer
    pub fn set_layer_weights(&mut self, layer: usize, weights: Matrix) {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].0 = weights;
    }

    /// Returns the weights of the given layer
    pub fn layer_weights(&self, layer: usize) -> Matrix {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].0.clone()
    }

    /// Sets the biases of the given layer
    pub fn set_layer_biases(&mut self, layer: usize, biases: Matrix) {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].1 = biases;
    }

    /// Returns the biases of the given layer
    pub fn layer_biases(&self, layer: usize) -> Matrix {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].1.clone()
    }

    /// Returns the number of hidden layers
    pub fn hidden_layers_size(&self) -> usize {
        self.hidden_layers.len()
    }

    /// Sets the learning rate of the network
    pub fn set_learning_rate(&mut self, alpha: f64) {
        self.leanring_rate = alpha;
    }

    /// Returns the learning rate of the network
    pub fn leanring_rate(&self) -> f64 {
        self.leanring_rate
    }

    /// Returns the output of the network for the given input. It doesn't consume the input
    ///
    /// ## Example
    /// ```
    /// // network creation and training
    /// // ...
    ///
    /// let prediction = network.forward_propagate(&[1, 3]); // Predict using the input [1, 3]
    /// ```
    pub fn forward_propagate(&mut self, input: &Vec<f64>) -> Vec<f64> {
        if !self.compiled {
            self.compile();
        }

        self.activation_matrices.clear();

        let (weights, biases) = &self.layer_matrices[0];
        let mut output: Vec<f64> = weights.dot_vec(input);
        output = output.add(&biases.to_vec());

        let update_weights = |weights: Vec<f64>, activation: ActivationType| match activation {
            ActivationType::SoftMax => {
                let out_clone = weights.clone();
                weights
                    .into_iter()
                    .map(|x| softmax(x, &out_clone))
                    .collect()
            }
            _ => weights
                .into_iter()
                .map(|x| match activation {
                    ActivationType::Sigmoid => sigm(x),
                    ActivationType::Tanh => tanh(x),
                    ActivationType::ArcTanh => arc_tanh(x),
                    ActivationType::Relu => relu(x),
                    ActivationType::LeakyRelu => leaky_relu(x),
                    ActivationType::SoftMax => {
                        panic!("Soft max should be handled before this function")
                    }
                    ActivationType::SoftPlus => softplus(x),
                })
                .collect(),
        };

        output = update_weights(output, self.activation.clone());

        self.activation_matrices
            .push(Matrix::from_vec(output.clone(), output.len(), 1));

        for i in 1..self.layer_matrices.len() {
            let (weights, biases) = &self.layer_matrices[i];

            output = weights.dot_vec(&output);
            output = output.add(&biases.to_vec());

            output = update_weights(output, self.activation.clone());

            self.activation_matrices
                .push(Matrix::from_vec(output.clone(), output.len(), 1));
        }

        output
    }

    /// Trains the network with the given input and target output.
    ///
    /// ## Example
    /// ```
    /// // network creation
    /// // ...
    ///
    /// let input = vec![1, 3];
    /// let target = vec![0.5];
    ///
    /// network.back_propagate(&input, &target);
    /// ```
    pub fn back_propagate(&mut self, input: &Vec<f64>, target: &Vec<f64>) -> Vec<f64>{
        let output = self.forward_propagate(input);

        let mut delta_weights: Vec<Matrix> = vec![];
        let mut delta_biases: Vec<Matrix> = vec![];

        // first iteration is calculated with the predicted output

        let mut error = vec![0.0; output.len()];
        let mut loss = vec![0.0; output.len()];
        for i in 0..output.len() {
            error[i] = target[i] - output[i];
            loss[i] = error[i].powi(2);
        }

        let mut d_z = vec![0.0; output.len()];

        for i in 0..output.len() {
            match self.activation {
                ActivationType::Sigmoid => d_z[i] = der_sigm(output[i]),
                ActivationType::Tanh => d_z[i] = der_tanh(output[i]),
                ActivationType::ArcTanh => d_z[i] = der_arc_tanh(output[i]),
                ActivationType::Relu => d_z[i] = der_relu(output[i]),
                ActivationType::LeakyRelu => d_z[i] = der_leaky_relu(output[i]),
                ActivationType::SoftMax => d_z[i] = der_softmax(output[i], &output),
                ActivationType::SoftPlus => d_z[i] = der_softplus(output[i]),
            }
        }

        d_z = d_z
            .into_par_iter()
            .enumerate()
            .map(|(i, x)| -2. * x * error[i])
            .collect();

        let d_w = Matrix::from_vec(d_z.clone(), d_z.len(), 1)
            .dot(&self.activation_matrices[self.activation_matrices.len() - 2].transpose());
        delta_weights.push(d_w);

        let d_b = Matrix::from_vec(d_z.clone(), d_z.len(), 1);
        delta_biases.push(d_b);

        for i in (2..self.layer_matrices.len()).rev() {
            let d_a = self.layer_matrices[i].0.transpose().dot(&Matrix::from_vec(
                d_z.clone(),
                d_z.len(),
                1,
            ));

            match self.activation {
                ActivationType::Sigmoid => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_sigm(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
                ActivationType::Tanh => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_tanh(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
                ActivationType::ArcTanh => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_arc_tanh(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
                ActivationType::Relu => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_relu(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
                ActivationType::LeakyRelu => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_leaky_relu(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
                ActivationType::SoftMax => {
                    d_z = d_a
                        .to_vec()
                        .into_iter()
                        .map(|x| x * der_softmax(x, &d_a.data))
                        .collect()
                }
                ActivationType::SoftPlus => {
                    d_z = d_a
                        .data
                        .into_iter()
                        .map(|x| x * der_softplus(self.activation_matrices[i - 1].get(0, 0)))
                        .collect()
                }
            }

            let d_w = Matrix::from_vec(d_z.clone(), d_z.len(), 1)
                .dot(&self.activation_matrices[i - 2].transpose());
            delta_weights.push(d_w);

            let d_b = Matrix::from_vec(d_z.clone(), d_z.len(), 1);
            delta_biases.push(d_b);
        }

        // final iteration is calculated with the input layer
        let d_a =
            self.layer_matrices[1]
                .0
                .transpose()
                .dot(&Matrix::from_vec(d_z.clone(), d_z.len(), 1));

        match self.activation {
            ActivationType::Sigmoid => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_sigm(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
            ActivationType::Tanh => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_tanh(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
            ActivationType::ArcTanh => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_arc_tanh(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
            ActivationType::Relu => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_relu(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
            ActivationType::LeakyRelu => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_leaky_relu(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
            ActivationType::SoftMax => {
                d_z = d_a
                    .to_vec()
                    .into_iter()
                    .map(|x| x * der_softmax(x, &d_a.data))
                    .collect()
            }
            ActivationType::SoftPlus => {
                d_z = d_a
                    .data
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| x * der_softplus(self.activation_matrices[0].get(i, 0)))
                    .collect()
            }
        }

        let d_w = Matrix::from_vec(d_z.clone(), d_z.len(), 1)
            .dot(&Matrix::from_vec(input.clone(), input.len(), 1).transpose());
        delta_weights.push(d_w);

        let d_b = Matrix::from_vec(d_z.clone(), d_z.len(), 1);
        delta_biases.push(d_b);

        for i in 0..self.layer_matrices.len() {
            self.layer_matrices[i].0 = self.layer_matrices[i].0.sub(
                &delta_weights[self.layer_matrices.len() - 1 - i].scalar_mul(self.leanring_rate),
            );
            self.layer_matrices[i].1 = self.layer_matrices[i].1.sub(
                &delta_biases[self.layer_matrices.len() - 1 - i].scalar_mul(self.leanring_rate),
            );
        }

        loss
    }
}

/// Formats the network to be printed
impl Display for Network {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut output = String::new();
        output.push_str(&format!("Inputs: {}\n", self.inputs.size));
        output.push_str(&format!("Outputs: {}\n", self.outputs.size));
        output.push_str(&format!("Hidden layers: {}\n", self.hidden_layers.len()));
        output.push_str(&format!("Activation: {:?}\n", self.activation));
        output.push_str(&format!("Weights:\n"));
        output.push_str(&format!("---------------------\n"));
        for (i, (weights, biases)) in self.layer_matrices.iter().enumerate() {
            output.push_str(&format!("Layer {}:\n", i));
            output.push_str(&format!("Weights:\n{}", weights));
            output.push_str(&format!("Biases:\n{}", biases));
            output.push_str(&format!("---------------------\n"));
        }
        write!(f, "{}", output)
    }
}
