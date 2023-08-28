//! Activation functions and their derivatives.
//!
//! The activation functions are used to determine the output of a neuron and to compute the back-propagation gradient.

use serde::{Deserialize, Serialize};

/// Determine types of activation functions contained in this module.
/// >   The network automaticaly uses the correct derivative when propagating
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ArcTanh,
    Relu,
    LeakyRelu,
    SoftMax,
    SoftPlus,
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

pub fn arc_tanh(x: f64) -> f64 {
    x.atan()
}

pub fn der_arc_tanh(x: f64) -> f64 {
    1.0 / (1.0 + x.powi(2))
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

pub fn leaky_relu(x: f64) -> f64 {
    if x <= 0.0 {
        0.01 * x
    } else {
        x
    }
}

pub fn der_leaky_relu(x: f64) -> f64 {
    if x <= 0.0 {
        0.01
    } else {
        1.0
    }
}

pub fn softmax(x: f64, total: &Vec<f64>) -> f64 {
    x.exp() / total.iter().map(|x| x.exp()).sum::<f64>()
}

pub fn softmax_array<const SIZE: usize>(x: f64, total: &[f64; SIZE]) -> f64 {
    x.exp() / total.iter().map(|x| x.exp()).sum::<f64>()
}

pub fn der_softmax(x: f64, total: &Vec<f64>) -> f64 {
    softmax(x, total) * (1.0 - softmax(x, total))
}

pub fn softplus(x: f64) -> f64 {
    x.ln_1p().exp()
}

pub fn der_softplus(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
