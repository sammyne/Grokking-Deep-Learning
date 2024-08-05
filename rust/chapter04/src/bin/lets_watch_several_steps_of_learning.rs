fn main() {
    let (mut weight, goal_pred, input) = (0.0, 0.8f64, 1.1f64);

    for _ in 0..4 {
        println!("-----\nWeight:{weight}");
        let pred = input * weight;
        let error = (pred - goal_pred).powi(2);
        let delta = pred - goal_pred;
        let weight_delta = delta * input;
        weight -= weight_delta;
        println!("Error:{error} Prediction:{pred}");
        println!("Delta:{delta} Weight Delta:{weight_delta}");
    }
}

// weight, goal_pred, input = (0.0, 0.8, 1.1)

// for iteration in range(4):
//     print("-----\nWeight:" + str(weight))
//     pred = input * weight
//     error = (pred - goal_pred) ** 2
//     delta = pred - goal_pred
//     weight_delta = delta * input
//     weight = weight - weight_delta
//     print("Error:" + str(error) + " Prediction:" + str(pred))
//     print("Delta:" + str(delta) + " Weight Delta:" + str(weight_delta))
