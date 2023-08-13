use rand::random;
use rayon::{prelude::*, vec};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::{self, Display, Formatter};
use std::{f64, result};

/// Determine types of activation functions contained in this module.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    Relu,
}

pub fn sigm(x: f64) -> f64 {
    1.0 / (1.0 + x.exp())
}
pub fn der_sigm(x: f64) -> f64 {
    sigm(x) * (1.0 - sigm(x))
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn der_tanh(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

pub fn relu(x: f64) -> f64 {
    f64::max(0.0, x)
}

pub fn der_relu(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

pub trait linear_algebra {
    fn add(&self, other: &Self) -> Self;
}

impl linear_algebra for Vec<f64> {
    fn add(&self, other: &Self) -> Self {
        debug_assert!(self.len() == other.len());
        let mut result = vec![0.0; self.len()];
        for i in 0..self.len() {
            result[i] = self[i] + other[i];
        }
        result
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0 && cols > 0);
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn from_vec(vec: Vec<f64>, rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0 && cols > 0 && rows * cols == vec.len());
        Matrix {
            data: vec,
            rows,
            cols,
        }
    }

    fn new_identity(size: usize) -> Self {
        debug_assert!(size > 0);
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Matrix {
            data,
            rows: size,
            cols: size,
        }
    }

    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[self.cols * row + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[self.cols * row + col] = value;
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn dot(&self, other: &Self) -> Matrix {
        debug_assert!(self.cols() == other.rows());

        let mut result = Matrix::new(self.rows(), other.cols());

        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = 0.0;
                for k in 0..self.cols() {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    fn dot_vec(&self, other: &Vec<f64>) -> Vec<f64> {
        debug_assert!(self.cols() == other.len());

        let mut result = vec![0.0; self.rows()];
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other[j];
                result[i] += a * b;
            }
        }
        result
    }

    fn scalar_mul(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(i, j, self.get(i, j) * scalar);
            }
        }
        result
    }

    fn add(&self, other: &Self) -> Matrix {
        debug_assert!(self.rows() == other.rows() && self.cols() == other.cols());

        let mut result = Matrix::new(self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }

    fn add_vec(&self, other: &Vec<f64>) -> Matrix {
        debug_assert!(self.rows() == other.len());

        let mut result = Matrix::new(self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(i, j, self.get(i, j) + other[i]);
            }
        }
        result
    }

    fn sub(&self, other: &Self) -> Matrix {
        debug_assert!(self.rows() == other.rows() && self.cols() == other.cols());

        let mut result = Matrix::new(self.rows(), self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
        result
    }

    fn to_vec(&self) -> Vec<f64> {
        debug_assert!(self.cols() == 1);
        self.data.clone()
    }

    fn get_value(&self) -> Option<f64> {
        if self.rows() == 1 && self.cols() == 1 {
            Some(self.get(0, 0))
        } else {
            None
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut output = String::new();
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                output.push_str(&format!("{} ", self.get(i, j)));
            }
            output.push_str("\n");
        }
        write!(f, "{}", output)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Layer {
    size: usize,
    bias: Vec<f64>,
}

impl Layer {
    fn new(size: usize) -> Self {
        Layer {
            size,
            bias: (0..size).into_par_iter().map(|_| random::<f64>()).collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Network {
    inputs: Layer,                         // number of neurons in input layer
    outputs: Layer,                        // number of neurons in output layer
    hidden_layers: Vec<Layer>, // number of hidden layers (each layer has a number of neurons)
    layer_matrices: Vec<(Matrix, Matrix)>, // (weights, biases)
    activation_matrices: Vec<Matrix>,
    activation: ActivationType, // activation function
    leanring_rate: f64,
}

impl Network {
    fn new(
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
        }
    }

    fn empty_network(
        inputs: usize,
        outputs: usize,
        activation_func: ActivationType,
        alpha: f64,
    ) -> Self {
        Network {
            inputs: Layer::new(inputs),
            outputs: Layer::new(outputs),
            hidden_layers: vec![],
            layer_matrices: vec![],
            activation_matrices: vec![],
            activation: activation_func,
            leanring_rate: alpha,
        }
    }

    fn add_hidden_layer(&mut self, layer: Layer) {
        self.hidden_layers.push(layer);
    }

    fn add_hidden_layer_with_size(&mut self, size: usize) {
        self.hidden_layers.push(Layer::new(size));
    }

    fn compile(&mut self) {
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
    }

    fn set_activation(&mut self, activation: ActivationType) {
        self.activation = activation;
    }

    fn set_layer_weights(&mut self, layer: usize, weights: Matrix) {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].0 = weights;
    }

    fn set_layer_biases(&mut self, layer: usize, biases: Matrix) {
        debug_assert!(layer < self.layer_matrices.len());
        self.layer_matrices[layer].1 = biases;
    }

    fn forward_propagate(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.activation_matrices.clear();

        let (weights, biases) = &self.layer_matrices[0];
        let mut output: Vec<f64> = weights.dot_vec(input);
        output = output.add(&biases.to_vec());

        output = output
            .into_par_iter()
            .map(|x| match self.activation {
                ActivationType::Sigmoid => sigm(x),
                ActivationType::Tanh => tanh(x),
                ActivationType::Relu => relu(x),
            })
            .collect();

        self.activation_matrices
            .push(Matrix::from_vec(output.clone(), output.len(), 1));

        for i in 1..self.layer_matrices.len() {
            let (weights, biases) = &self.layer_matrices[i];

            output = weights.dot_vec(&output);
            output = output.add(&biases.to_vec());

            output = output
                .into_par_iter()
                .map(|x| match self.activation {
                    ActivationType::Sigmoid => sigm(x),
                    ActivationType::Tanh => tanh(x),
                    ActivationType::Relu => relu(x),
                })
                .collect();

            self.activation_matrices
                .push(Matrix::from_vec(output.clone(), output.len(), 1));
        }

        output
    }

    fn back_propagate(&mut self, input: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
        let mut output = self.forward_propagate(input);

        let mut d_a: Matrix = Matrix::new(1, 1);

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
                ActivationType::Relu => d_z[i] = der_relu(output[i]),
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
            d_a = self.layer_matrices[i].0.transpose().dot(&Matrix::from_vec(
                d_z.clone(),
                d_z.len(),
                1,
            ));

            d_z = d_a
                .to_vec()
                .into_par_iter()
                .map(|x| {
                    x * match self.activation {
                        ActivationType::Sigmoid => {
                            der_sigm(self.activation_matrices[i - 1].get(0, 0))
                        }
                        ActivationType::Tanh => der_tanh(self.activation_matrices[i - 1].get(0, 0)),
                        ActivationType::Relu => der_relu(self.activation_matrices[i - 1].get(0, 0)),
                    }
                })
                .collect();

            let d_w = Matrix::from_vec(d_z.clone(), d_z.len(), 1)
                .dot(&self.activation_matrices[i - 2].transpose());
            delta_weights.push(d_w);

            let d_b = Matrix::from_vec(d_z.clone(), d_z.len(), 1);
            delta_biases.push(d_b);
        }

        // final iteration is calculated with the input layer
        dbg!(d_a.clone());

        d_a =
            self.layer_matrices[1]
                .0
                .transpose()
                .dot(&Matrix::from_vec(d_z.clone(), d_z.len(), 1));

        dbg!(d_a.clone());
        d_z = d_a
            .to_vec()
            .into_par_iter()
            .enumerate()
            .map(|(i, x)| {
                x * match self.activation {
                    ActivationType::Sigmoid => der_sigm(self.activation_matrices[0].get(i, 0)),
                    ActivationType::Tanh => der_tanh(self.activation_matrices[0].get(i, 0)),
                    ActivationType::Relu => der_relu(self.activation_matrices[0].get(i, 0)),
                }
            })
            .collect();

        dbg!(d_z.clone());

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

        // let matrix_index = self.layer_matrices.len() - 1;
        // self.layer_matrices[matrix_index].0 = self.layer_matrices[matrix_index].0.sub(&delta_weights.scalar_mul(self.leanring_rate));

        // self.layer_matrices[matrix_index].1 = self.layer_matrices[matrix_index].1.sub(&delta_biases.scalar_mul(self.leanring_rate));

        d_z
    }
}

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

fn main() {
    let mut network = Network::empty_network(3, 1, ActivationType::Relu, 0.005);

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
    let error = network.back_propagate(&input, &vec![9.0]);
    let new_prediction = network.forward_propagate(&input);

    // println!("{}", network);
    println!("{:?}", prediction);
    println!("{:?}", error);
    println!("{:?}", new_prediction);
}
