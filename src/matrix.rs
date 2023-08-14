//! # Matrix
//!
//! A matrix implementation that supports basic matrix operations. The heavy functions are all parallelized.
//!
//! > This Matrix implementation is **NOT** meant to be used as a general purpose matrix library. It is only meant to be used for the neural network library for now.
//!
//! > A stack-based matrix implementation is planned for the future.
//!

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::f64;
use std::fmt::{self, Display, Formatter};

/// A matrix implementation that supports basic matrix operations. The heavy functions are all parallelized.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Creates a new matrix with the given dimensions.
    ///
    /// ## Example
    /// ```
    /// let matrix = Matrix::new(3, 2);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0 && cols > 0);
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a new matrix with the given dimensions and fills it with the given value.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::matrix::*;
    ///
    /// let matrix = Matrix::from_vec(
    ///     vec![0.03, 0.62, 0.85, 0.60, 0.62, 0.64],
    ///     3,
    ///     2,);
    ///
    /// assert_eq!(matrix.get(0, 1), 0.62);
    /// assert_eq!(matrix.rows(), 3);
    /// assert_eq!(matrix.cols(), 2);
    /// ```
    ///
    pub fn from_vec(vec: Vec<f64>, rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0 && cols > 0 && rows * cols == vec.len());
        Matrix {
            data: vec,
            rows,
            cols,
        }
    }

    /// Creates a new matrix from the given JSON string.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::matrix::*;
    ///
    /// let matrix = Matrix::from_json(
    ///    r#"{
    ///       "data": [
    ///         0.03,
    ///         0.62,
    ///         0.85,
    ///         0.60,
    ///         0.62,
    ///         0.64
    ///         ],
    ///         "rows": 3,
    ///         "cols": 2
    ///     }"#);
    ///
    /// assert_eq!(matrix.get(0, 1), 0.62);
    /// ```
    ///
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap()
    }

    /// transforms the matrix into a JSON string.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::matrix::*;
    ///
    /// let matrix = Matrix::from_vec(
    ///     vec![0.03, 0.62, 0.85, 0.60, 0.62, 0.64],
    ///     3,
    ///     2,);
    ///
    /// assert_eq!(matrix.to_json(), r#"{"data":[0.03,0.62,0.85,0.6,0.62,0.64],"rows":3,"cols":2}"#);
    /// ```
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    /// Saves the matrix to the given path.
    ///
    /// ## Example
    /// ```
    /// use fast_neural_network::matrix::*;
    ///
    /// let matrix = Matrix::from_vec(
    ///     vec![0.03, 0.62, 0.85, 0.60, 0.62, 0.64],
    ///     3,
    ///     2,);
    ///
    /// matrix.save("matrix.json");
    /// ```
    pub fn save(&self, path: &str) {
        std::fs::write(path, self.to_json()).unwrap();
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Gets the value at the given row and column.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[self.cols * row + col]
    }

    /// Sets the value at the given row and column.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[self.cols * row + col] = value;
    }

    /// Returns the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Multiplies the matrix with the given matrix.
    pub fn dot(&self, other: &Self) -> Matrix {
        debug_assert!(self.cols() == other.rows());

        let mut result = Matrix::new(self.rows(), other.cols());

        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let sum = (0..self.cols())
                    .into_par_iter()
                    .map(|k| self.get(i, k) * other.get(k, j))
                    .sum();
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Multiplies the matrix with the given vector.
    pub fn dot_vec(&self, other: &Vec<f64>) -> Vec<f64> {
        debug_assert!(self.cols() == other.len());

        (0..self.rows())
            .into_par_iter()
            .map(|i| {
                (0..self.cols())
                    .into_par_iter()
                    .map(|j| self.get(i, j) * other[j])
                    .sum()
            })
            .collect()
    }

    /// Multiplies the matrix with the given vector.
    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let mut result = Matrix {
            data: self.data.clone(),
            rows: self.rows(),
            cols: self.cols(),
        };

        result.data.par_iter_mut().for_each(|x| *x *= scalar);

        result
    }

    /// Subptracts the given matrix from the matrix.
    pub fn sub(&self, other: &Self) -> Matrix {
        debug_assert!(self.rows() == other.rows() && self.cols() == other.cols());

        Matrix {
            data: (0..self.rows() * self.cols())
                .into_par_iter()
                .map(|i| self.data[i] - other.data[i])
                .collect(),
            rows: self.rows(),
            cols: self.cols(),
        }
    }

    /// transforms the matrix into a vector.
    pub fn to_vec(&self) -> Vec<f64> {
        debug_assert!(self.cols() == 1);
        self.data.clone()
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
