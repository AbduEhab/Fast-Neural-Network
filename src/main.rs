use fast_neural_network::{activation::*, neural_network::*};
use ndarray::*;

fn main() {
    let mut network = Network::new(2, 1, ActivationType::Tanh, 0.005);

    network.add_hidden_layer_with_size(2);
    network.compile();

    // Let's create a dataset
    let mut dataset: Vec<(ndarray::Array1<f64>, ndarray::Array1<f64>)> = Vec::new();

    dataset.push((array!(0., 0.), array!(0.)));
    dataset.push((array!(1., 0.), array!(1.)));
    dataset.push((array!(0., 1.), array!(1.)));
    dataset.push((array!(1., 1.), array!(0.)));

    network.train(&dataset, 50_000, 5_000);

    let mut res;

    // Let's check the result
    for i in 0..dataset.len() {
        res = network.forward(&dataset[i].0);
        let d = &dataset[i];
        println!(
            "for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
            d.0[0], d.0[1], d.1[0], res
        );
    }

    network.save("network.json");

    // let mut network = Network::load("network.json");

    // println!("{:?}", network.predict(&input));
}
