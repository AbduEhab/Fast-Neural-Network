use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use fast_neural_network::matrix::Matrix;

trait Bench {
    fn seq_dot(&self, other: &Self) -> Matrix;
    fn par_dot(&self, other: &Self) -> Matrix;
}

impl Bench for Matrix {
    fn seq_dot(&self, other: &Self) -> Matrix {
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

    fn par_dot(&self, other: &Self) -> Matrix {
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
}

const MATRIX_SIZE: usize = 10;

fn matrix_sequential_dot(c: &mut Criterion) {
    let mut matrix_a = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);
    let mut matrix_b = Matrix::new(MATRIX_SIZE, MATRIX_SIZE);

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a.set(i, j, i as f64 + j as f64);
            matrix_b.set(i, j, i as f64 + j as f64);
        }
    }

    c.bench_function("matrix sequential dot", |b| b.iter(|| matrix_a.seq_dot(&matrix_b)));
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

    c.bench_function("matrix parallel dot", |b| b.iter(|| matrix_a.par_dot(&matrix_b)));
}

criterion_group!(benches, matrix_sequential_dot, matrix_parallel_dot);
criterion_main!(benches);
