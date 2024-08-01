fn main() {
    let mut weight = 0.5;

    const INPUT: f64 = 0.5;
    const GOAL_PREDICTION: f64 = 0.8;

    const STEP_AMOUNT: f64 = 0.001;

    for _ in 0..1101 {
        let prediction = INPUT * weight;
        let error = (prediction - GOAL_PREDICTION).powi(2);

        println!("Error: {} Prediction: {}", error, prediction);

        let up_prediction = INPUT * (weight + STEP_AMOUNT);
        let up_error = (GOAL_PREDICTION - up_prediction).powi(2);

        let down_prediction = INPUT * (weight - STEP_AMOUNT);
        let down_error = (GOAL_PREDICTION - down_prediction).powi(2);

        if down_error < up_error {
            weight -= STEP_AMOUNT;
        }

        if down_error > up_error {
            weight += STEP_AMOUNT
        }
    }
}
