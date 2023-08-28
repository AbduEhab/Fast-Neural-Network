use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use fast_neural_network::{activation::*, matrix::*, neural_network::*};

trait Bench {
    fn seq_dot(&self, other: &Self) -> Matrix;
    fn par_dot(&self, other: &Self) -> Matrix;
    fn seq_transpose(&self) -> Matrix;
    fn par_transpose(&self) -> Matrix;
}

impl Bench for Matrix {
    fn par_dot(&self, other: &Self) -> Matrix {
        debug_assert!(self.cols() == other.rows());

        Matrix {
            data: (0..self.rows())
                .into_par_iter()
                .flat_map(|i| {
                    (0..other.cols())
                        .into_par_iter()
                        .map(|j| {
                            (0..self.cols())
                                .into_par_iter()
                                .map(|k| {
                                    self.data[i * self.cols + k] * other.data[k * other.cols + j]
                                })
                                .sum()
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<f64>>(),
            rows: self.rows(),
            cols: other.cols(),
        }
    }

    fn seq_dot(&self, other: &Self) -> Matrix {
        debug_assert!(self.cols() == other.rows());

        let mut result = Matrix::new(self.rows(), other.cols());

        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = 0.0;
                for k in 0..self.cols() {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    fn seq_transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols(), self.rows());

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    fn par_transpose(&self) -> Matrix {
        Matrix {
            data: (0..self.cols())
                .into_par_iter()
                .flat_map(|i| {
                    (0..self.rows())
                        .into_par_iter()
                        .map(|j| self.data[j * self.cols + i])
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<f64>>(),
            rows: self.cols(),
            cols: self.rows(),
        }
    }
}

const MATRIX_SIZE: usize = 100;

fn matrix_sequential_dot(c: &mut Criterion) {
    let mut matrix_a = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);
    let mut matrix_b = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a.set(i, j, i as f64 + j as f64);
            matrix_b.set(i, j, i as f64 + j as f64);
        }
    }

    c.bench_function("matrix sequential dot", |b| {
        b.iter(|| matrix_a.seq_dot(&matrix_b))
    });
}

fn matrix_parallel_dot(c: &mut Criterion) {
    let mut matrix_a = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);
    let mut matrix_b = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a.set(i, j, i as f64 + j as f64);
            matrix_b.set(i, j, i as f64 + j as f64);
        }
    }

    c.bench_function("matrix parallel dot", |b| {
        b.iter(|| matrix_a.par_dot(&matrix_b))
    });
}

fn matrix_sequential_transpose(c: &mut Criterion) {
    let mut matrix_a = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a.set(i, j, i as f64 + j as f64);
        }
    }

    c.bench_function("matrix sequential transpose", |b| {
        b.iter(|| matrix_a.seq_transpose())
    });
}

fn matrix_parallel_transpose(c: &mut Criterion) {
    let mut matrix_a = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a.set(i, j, i as f64 + j as f64);
        }
    }

    c.bench_function("matrix parallel transpose", |b| {
        b.iter(|| matrix_a.par_transpose())
    });
}

fn par_prop(c: &mut Criterion) {
    let mut network = Network::new(3, 1, ActivationType::Relu, 0.005);

    for _ in 0..64 {
        network.add_hidden_layer_with_size(64);
    }

    network.compile();

    c.bench_function("par_prop", |b| {
        b.iter(|| network.back_propagate(&vec![2., 1., -1.], &vec![9.0]))
    });
}

criterion_group!(
    benches,
    matrix_sequential_dot,
    matrix_parallel_dot,
    // matrix_sequential_transpose,
    // matrix_parallel_transpose,
    // par_prop
);
criterion_main!(benches);
