use ndarray::{Array, array, s};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn main() {
    const ALPHA: f64 = 0.2;
    const HIDDEN_SIZE: usize = 4;

    let streetlights = array![[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]].map(|v| *v as f64);

    let walk_vs_stop = array![[1, 1, 0, 0]].map(|v| *v as f64).t().to_owned();

    let mut r = SmallRng::seed_from_u64(1);
    let distrib = Uniform::new(0.0, 1.0);

    let mut weights_0_1 =
        2.0f64 * Array::random_using((3, HIDDEN_SIZE), distrib.clone(), &mut r) - 1.0;
    let mut weights_1_2 = 2.0f64 * Array::random_using((HIDDEN_SIZE, 1), distrib, &mut r) - 1.0;

    for j in 0..60 {
        let mut layer_2_error = 0.0;
        for i in 0..streetlights.nrows() {
            let layer_0 = streetlights.slice(s![i..(i + 1), ..]);
            let layer_1 = layer_0.dot(&weights_0_1).map(|v| relu(*v));
            let layer_2 = layer_1.dot(&weights_1_2);

            let ws = walk_vs_stop.slice(s![i..(i + 1), ..]);

            layer_2_error += (layer_2.clone() - ws).pow2().sum();

            let layer_2_delta = layer_2 - ws.to_owned();
            let layer_1_delta =
                layer_2_delta.dot(&weights_1_2.t()) * layer_1.map(|v| relu2deriv(*v));

            weights_1_2 = weights_1_2 - layer_1.t().dot(&layer_2_delta) * ALPHA;
            weights_0_1 = weights_0_1 - layer_0.t().dot(&layer_1_delta) * ALPHA;
        }

        if j % 10 == 9 {
            println!("Error: {}", layer_2_error);
        }
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu2deriv(output: f64) -> f64 {
    if output > 0.0 { 1.0 } else { 0.0 }
}
