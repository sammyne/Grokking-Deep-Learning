fn main() {
    const GOAL_PRED: f64 = 0.8;
    const INPUT: f64 = 0.5;

    let mut weight = 0.5;

    for _ in 0..20 {
        let pred = INPUT * weight;
        let error = (pred - GOAL_PRED).powi(2);
        let direction_and_amount = (pred - GOAL_PRED) * INPUT;
        weight -= direction_and_amount;

        println!("Error:{} Prediction:{}", error, pred);
    }
}
