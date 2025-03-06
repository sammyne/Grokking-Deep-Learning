use ndarray::{Array, Order, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn main() {
    //const ALPHA: f64 = 0.2;
    const HIDDEN_SIZE: usize = 4;

    let streetlights = array![[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]].map(|v| *v as f64);

    //let walk_vs_stop = array![[1, 1, 0, 0]].map(|v| *v as f64).t().to_owned();

    let weights_0_1 = 2.0f64 * Array::random((3, HIDDEN_SIZE), Uniform::new(0.0, 1.0)) - 1.0;
    let weights_1_2 = 2.0f64 * Array::random((HIDDEN_SIZE, 1), Uniform::new(0.0, 1.0)) - 1.0;

    let layer_0 = streetlights
        .row(0)
        .into_shape_with_order(((1, 3), Order::ColumnMajor))
        .expect("reshape layer1")
        .to_owned();
    let layer_1 = layer_0.dot(&weights_0_1).map(|v| relu(*v));
    let _layer_2 = layer_1.dot(&weights_1_2);
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}
