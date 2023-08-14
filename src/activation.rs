//! Activation functions and their derivatives.
//! 
//! The activation functions are used to determine the output of a neuron and to compute the back-propagation gradient.

use serde::{Serialize, Deserialize};

/// Determine types of activation functions contained in this module.
/// >   The network automaticaly uses the correct derivative when propagating
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