//! # Neural Network
//!
//! This is the neural network module. It contains the `Network` struct and the `ActivationType` enum. This is the heart of the crate.
//!

use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use ndarray::*;
use rand::random;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::f64;
use std::fmt::{self, Display, Formatter, Write};

use crate::activation::*;

/// The underlying layer struct for the heap based network.
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

/// The main heap based neural network struct.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Network {
    inputs: Layer,             // number of neurons in input layer
    outputs: Layer,            // number of neurons in output layer
    hidden_layers: Vec<Layer>, // number of hidden layers (each layer has a number of neurons)
    layer_matrices: Vec<(ndarray::Array2<f64>, ndarray::Array2<f64>)>, // (weights, biases)
    activation_matrices: Vec<ndarray::Array2<f64>>,
    activation: ActivationType, // activation function
    learning_rate: f64,
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
    /// assert_eq!(network.learning_rate(), 0.005);
    /// ```
    pub fn new(inputs: usize, outputs: usize, activation_func: ActivationType, alpha: f64) -> Self {
        Network {
            inputs: Layer::new(inputs),
            outputs: Layer::new(outputs),
            hidden_layers: vec![],
            layer_matrices: vec![],
            activation_matrices: vec![],
            activation: activation_func,
            learning_rate: alpha,
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
    /// assert_eq!(network.learning_rate(), 0.005);
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
            learning_rate: alpha,
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
    /// Panics if any of the dimentions is 0 or if no hidden layers are present.
    pub fn compile(&mut self) {
        assert!(self.inputs.size > 0);
        assert!(self.outputs.size > 0);

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
            let weights =
                Array2::from_shape_vec((layers[i + 1].size, layers[i].size), weights).unwrap();
            let biases = Array2::from_shape_vec((layers[i + 1].size, 1), biases).unwrap();
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
    pub fn set_layer_weights(&mut self, layer: usize, weights: ndarray::Array2<f64>) {
        assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].0 = weights;
    }

    /// Returns the weights of the given layer
    pub fn layer_weights(&self, layer: usize) -> ndarray::Array2<f64> {
        assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].0.clone()
    }

    /// Sets the biases of the given layer
    pub fn set_layer_biases(&mut self, layer: usize, biases: ndarray::Array2<f64>) {
        assert!(layer < self.layer_matrices.len());
        assert!(biases.len() == self.layer_matrices[layer].1.len());

        self.layer_matrices[layer].1 = biases.t().to_owned();
    }

    /// Returns the biases of the given layer
    pub fn layer_biases(&self, layer: usize) -> ndarray::Array2<f64> {
        assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].1.clone()
    }

    /// Returns the number of hidden layers
    pub fn hidden_layers_size(&self) -> usize {
        self.hidden_layers.len()
    }

    /// Sets the learning rate of the network
    pub fn set_learning_rate(&mut self, alpha: f64) {
        self.learning_rate = alpha;
    }

    /// Returns the learning rate of the network
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn back_propagate_helper(
        &self,
        dz: &ndarray::Array2<f64>,
        a: &ndarray::Array2<f64>,
        weights: &ndarray::Array2<f64>,
    ) -> (
        ndarray::Array2<f64>,
        ndarray::Array2<f64>,
        ndarray::Array1<f64>,
    ) {
        let m = self.outputs.size as f64;

        let dw = dz.dot(&a.t()) / m;
        let db = &dz.column(0) / m;
        let da = weights.t().dot(dz);

        (
            Array::from_shape_vec((da.len(), 1), da.into_raw_vec()).unwrap(),
            dw,
            db,
        )
    }

    fn activate(&self, weights: &mut ndarray::Array2<f64>) {
        match self.activation {
            ActivationType::Sigmoid => weights.column_mut(0).into_iter().for_each(|x| {
                *x = sigm(*x);
            }),
            ActivationType::Tanh => weights.column_mut(0).into_iter().for_each(|x| {
                *x = tanh(*x);
            }),
            ActivationType::ArcTanh => weights.column_mut(0).into_iter().for_each(|x| {
                *x = arc_tanh(*x);
            }),
            ActivationType::Relu => weights.column_mut(0).into_iter().for_each(|x| {
                *x = relu(*x);
            }),
            ActivationType::LeakyRelu => weights.column_mut(0).into_iter().for_each(|x| {
                *x = leaky_relu(*x);
            }),
            ActivationType::ELU => weights.column_mut(0).into_iter().for_each(|x| {
                *x = elu(*x);
            }),
            ActivationType::Swish => weights.column_mut(0).into_iter().for_each(|x| {
                *x = swish(*x);
            }),
            ActivationType::SoftPlus => weights.column_mut(0).into_iter().for_each(|x| {
                *x = softplus(*x);
            }),
            ActivationType::SoftMax => {
                let w_col = weights.column(0).to_owned();
                weights.column_mut(0).into_iter().for_each(|x| {
                    *x = softmax(*x, &w_col);
                })
            }
        };
    }
    /// Predicts the output of the network for the given input.
    ///
    /// ## Example
    /// ```
    /// // ... imports here
    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    ///
    /// // ... training done here
    ///
    /// let prediction = network.forward(&array![2., 1., -1.]);
    pub fn forward(&mut self, input: &ndarray::Array1<f64>) -> ndarray::Array2<f64> {
        if !self.compiled {
            self.compile();
        }

        assert!(input.len() == self.inputs.size);

        self.activation_matrices.clear();

        let mut output = input.clone().into_shape((self.inputs.size, 1)).unwrap();

        for i in 0..self.layer_matrices.len() {
            let (weights, biases) = &self.layer_matrices[i];

            output = weights.dot(&output);
            output = output + biases;

            self.activate(&mut output);

            self.activation_matrices.push(output.clone());
        }

        output.into_shape((self.outputs.size, 1)).unwrap()
    }

    fn derivate(&self, mut array: ndarray::Array2<f64>) -> ndarray::Array1<f64> {
        let array_size = array.len();
        match self.activation {
            ActivationType::Sigmoid => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_sigm(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::Tanh => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_tanh(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::ArcTanh => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_arc_tanh(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::Relu => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_relu(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::LeakyRelu => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_leaky_relu(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::ELU => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_elu(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::Swish => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_swish(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::SoftPlus => {
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_softplus(*x);
                });
                array.into_shape(array_size).unwrap()
            }
            ActivationType::SoftMax => {
                let array_col = array.column(0).to_owned();
                array.column_mut(0).into_iter().for_each(|x| {
                    *x = der_softmax(*x, &array_col);
                });
                array.into_shape(array_size).unwrap()
            }
        }
    }

    /// Trains the network with the given training set for the given number of epochs.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::{activation::*, neural_network::*};
    /// use ndarray::*;
    ///
    /// let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);
    ///
    /// network.add_hidden_layer_with_size(4);
    /// network.add_hidden_layer_with_size(4);
    ///
    /// network.compile();
    ///
    /// network.train(&[(array![2., 1., -1.], array![9.])], 100, 100);
    /// ```
    pub fn train(
        &mut self,
        training_set: &[(ndarray::Array1<f64>, ndarray::Array1<f64>)],
        epochs: usize,
        decay_time: usize,
    ) {
        let m = MultiProgress::new();

        let outer = m.add(ProgressBar::new(epochs as u64));
        outer.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed}] {wide_bar:.cyan/blue} {pos:>7}/{len:7} {msg} {eta}",
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, weights: &mut dyn Write| {
                write!(weights, "eta: {:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-"),
        );

        let mut time_since_last_decay = 0;
        for _ in 0..epochs {
            time_since_last_decay += 1;

            let inner_pb = m.add(ProgressBar::new(training_set.len() as u64));
            inner_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] {bar:.green} {pos:>7}/{len:7} {eta}",
                    )
                    .unwrap()
                    .progress_chars("#>-")
                    .with_key("eta", |state: &ProgressState, weights: &mut dyn Write| {
                        write!(weights, "eta: {:.1}s", state.eta().as_secs_f64()).unwrap()
                    }),
            );

            if time_since_last_decay >= decay_time {
                self.learning_rate *= 0.95;
                time_since_last_decay = 0;
            }

            let mut element_counter = 0;

            for (input, target) in training_set {
                element_counter += 1;

                if element_counter % 100 == 0 {
                    inner_pb.inc(100);
                }

                let output = self.forward(input);

                let output = output.into_shape(self.outputs.size).unwrap();

                let mut z = target - &output;

                z.iter_mut().enumerate().for_each(|(i, x)| {
                    *x = *x
                        * -2.
                        * match self.activation {
                            ActivationType::Sigmoid => der_sigm(output[i]),
                            ActivationType::Tanh => der_tanh(output[i]),
                            ActivationType::ArcTanh => der_arc_tanh(output[i]),
                            ActivationType::Relu => der_relu(output[i]),
                            ActivationType::LeakyRelu => der_leaky_relu(output[i]),
                            ActivationType::ELU => der_elu(output[i]),
                            ActivationType::Swish => der_swish(output[i]),
                            ActivationType::SoftMax => der_softmax(output[i], &output),
                            ActivationType::SoftPlus => der_softplus(output[i]),
                        };
                });

                z.iter_mut().enumerate().for_each(|(i, z)| {
                    let dz: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = array![*z];

                    let layer_matrix_size = self.layer_matrices.len() - 1;

                    let a = &self.activation_matrices[layer_matrix_size - 1];
                    let (weights, biases) = &self.layer_matrices[layer_matrix_size];

                    let m = dz.len() as f64;

                    let weights = weights.row(i);

                    let dw = &dz * &a.t() / m;
                    let db = &dz / m;
                    let da = weights
                        .mapv(|v| v * dz[0])
                        .into_shape((a.len(), 1))
                        .unwrap();

                    let a_derived = self.derivate(a.clone());

                    let mut dz = Array::from_shape_vec(
                        (a_derived.len(), 1),
                        da.iter()
                            .zip(a_derived.iter())
                            .map(|(a, b)| a * b)
                            .collect::<Vec<f64>>(),
                    )
                    .unwrap();

                    let new_i_weights = (-&dw * self.learning_rate + weights)
                        .into_shape(weights.len())
                        .unwrap();
                    let new_i_biases = -&db * self.learning_rate + biases.row(i);

                    self.layer_matrices[layer_matrix_size]
                        .0
                        .row_mut(i)
                        .assign(&new_i_weights);
                    self.layer_matrices[layer_matrix_size]
                        .1
                        .row_mut(i)
                        .assign(&new_i_biases);

                    drop(new_i_weights);
                    drop(new_i_biases);

                    for j in (1..self.layer_matrices.len() - 1).rev() {
                        let dz_size = dz.len();

                        let (weights, biases) = &self.layer_matrices[j];

                        let a = &self.activation_matrices[j - 1];
                        let a_size = a.len();

                        let (da, dw, db) = self.back_propagate_helper(
                            &dz.into_shape((dz_size, 1)).unwrap(),
                            &a.clone().into_shape((a_size, 1)).unwrap(),
                            &weights,
                        );

                        dz = da.t().to_owned() * self.derivate(a.clone());

                        let n_weights = weights - &dw.mapv(|x| x * self.learning_rate);
                        let n_biases = biases
                            - &db
                                .mapv(|x| x * self.learning_rate)
                                .into_shape((db.len(), 1))
                                .unwrap();
                        self.layer_matrices[j] = (n_weights, n_biases);
                    }

                    let dz_size = dz.len();
                    let weights = &self.layer_matrices[0].0;
                    let a = input.clone().into_shape((input.len(), 1)).unwrap();
                    let (_, dw, db) = self.back_propagate_helper(
                        &dz.into_shape((dz_size, 1)).unwrap(),
                        &a,
                        &weights,
                    );

                    let (weights, biases) = &self.layer_matrices[0];
                    let n_weights = weights - &dw.mapv(|x| x * self.learning_rate);
                    let n_biases = biases
                        - &db
                            .mapv(|x| x * self.learning_rate)
                            .into_shape((db.len(), 1))
                            .unwrap();

                    self.layer_matrices[0] = (n_weights, n_biases);
                });
            }
            inner_pb.finish_and_clear();
            outer.inc(1);
        }
        outer.finish_and_clear();
        m.clear().unwrap();
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
