use ndarray::array;

fn main() {
    let mut weights = array![0.5f64, 0.48, -0.7];
    const ALPHA: f64 = 0.1;

    let streetlights = array![
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1]
    ]
    .map(|v| *v as f64);

    let walk_vs_stop = array![0, 1, 0, 1, 1, 0].map(|&v| v as f64);

    for _ in 0..40 {
        let mut error_for_all_lights = 0.0;
        for row_index in 0..walk_vs_stop.len() {
            //let input = streetlights[row_index].to_owned();
            let input = streetlights.row(row_index).to_owned();
            let goal_prediction = walk_vs_stop[row_index];

            let prediction = input.dot(&weights);

            let error = (goal_prediction - prediction).powi(2);
            error_for_all_lights += error;

            let delta = prediction - goal_prediction;
            weights = weights - (ALPHA * (input * delta));
            println!("Prediction: {prediction}");
        }
        println!("Error: {error_for_all_lights}\n");
    }
}
