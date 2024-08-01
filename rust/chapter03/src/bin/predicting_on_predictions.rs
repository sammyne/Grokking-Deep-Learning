use ndarray::{ArrayBase, OwnedRepr, ViewRepr};

fn main() {
    let ih_wgt = ndarray::arr2(&[
        [0.1, 0.2, -0.1], // hid[0]
        [-0.1, 0.1, 0.9], // hid[1]
        [0.1, 0.4, 0.1],  // hid[2]
    ]);
    let ih_wgt = ih_wgt.t();

    let hp_wgt: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = ndarray::arr2(&[
        // #hid[0] hid[1] hid[2]
        [0.3, 1.1, -0.3], // hurt?
        [0.1, 0.2, 0.0],  // win?
        [0.0, 1.3, 0.1],  // sad?
    ]);
    let hp_wgt = hp_wgt.t();

    let toes = [8.5, 9.5, 9.9, 9.0];
    let wlrec = [0.65, 0.8, 0.8, 0.9];
    let nfans = [1.2, 1.3, 0.5, 1.0];

    // # Input corresponds to every entry
    // # for the first game of the season.
    let input = [toes[0], wlrec[0], nfans[0]].to_vec();

    let weights = [ih_wgt, hp_wgt];

    let pred = neural_network(input, &weights);

    println!("{pred:?}");
}

fn neural_network(
    input: Vec<f64>,
    weights: &[ArrayBase<ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>; 2],
) -> Vec<f64> {
    let input: ArrayBase<OwnedRepr<_>, _> = input.into();

    input.dot(&weights[0]).dot(&weights[1]).to_vec()
}
