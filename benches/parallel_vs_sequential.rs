use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use fast_neural_network::{activation::*, neural_network::*};

use ndarray::*;

const MATRIX_SIZE: usize = 1000;

fn ndarray_dot(c: &mut Criterion) {
    let mut matrix_a = Array2::<f64>::zeros((MATRIX_SIZE, MATRIX_SIZE));
    let mut matrix_b = Array2::<f64>::zeros((MATRIX_SIZE, MATRIX_SIZE));

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            matrix_a[[i, j]] = 1. + i as f64 + j as f64;
            matrix_b[[i, j]] = 1. + i as f64 + j as f64;
        }
    }

    c.bench_function("ndarray dot", |b| b.iter(|| matrix_a.dot(&matrix_b)));
}

criterion_group!(
    benches,
    ndarray_dot,
);
criterion_main!(benches);
