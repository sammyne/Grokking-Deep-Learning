fn main() {
    let mut weight = 0.5;
    const GOAL_PRED: f64 = 0.8;
    const INPUT: f64 = 2.0;
    const ALPHA: f64 = 0.1;

    for _ in 0..20 {
        let pred = INPUT * weight;
        let error = (pred - GOAL_PRED).powi(2);
        let derivative = INPUT * (pred - GOAL_PRED);
        weight -= ALPHA * derivative;

        println!("Error:{error} Prediction:{pred}");
    }
}
