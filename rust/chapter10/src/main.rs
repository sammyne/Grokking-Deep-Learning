use mnist::{Mnist, MnistBuilder};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn main() {
    let ((x_train,y_train), (x_test,y_test)) = load_data();

    let images = x_train.to_shape((1000, 28*28))
    .expect("reshape x_train").map(|&v| v as f64) 
    /255.0;
    let labels = y_train;

    let one_hot_labels = {
        let mut v: Array2D = ArrayBase::zeros((labels.len(), 10));
        for (i,&l) in labels.iter().enumerate() {
            v[[i, l as usize]] = 1;
        }
        v
    };
    let labels = one_hot_labels;

    let test_images = x_test.to_shape((x_test.len(), 28*28)).expect("reshape x_test").map(|&v| v as f64)/255.0;
    let test_labels = {
        let mut v:  Array2D = ArrayBase::zeros((y_test.len(), 10));
        for (i,&l) in y_test.iter().enumerate() {
            v[[i,l as usize]] = 1;
        }
        v
    };

    const ALPHA:f64 = 2.0;
    const ITERATIONS: usize=300;
    const PIXELS_PER_IMAGE:usize = 28*28;
    const NUM_LABELS:usize = 10;
    const BATCH_SIZE:usize = 128;

    const INPUT_ROWS:usize = 28;
    const INPUT_COLS:usize = 28;

    const KERNEL_ROWS:usize = 3;
    const KERNEL_COLS:usize = 3;
    const NUM_KERNELS:usize = 16;

    const HIDDEN_SIZE:usize = (INPUT_ROWS-KERNEL_ROWS)*(INPUT_COLS-KERNEL_COLS)*NUM_KERNELS;

    let mut kernels = 0.02*Array::random((KERNEL_ROWS*KERNEL_COLS, NUM_KERNELS), Uniform::new(0., 1.))-0.01;

}

type Array1D = ArrayBase<OwnedRepr<u8>, Dim<[usize; 1]>>;
type Array2D = ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>;

fn load_data() -> ((Array1D, Array1D), (Array1D, Array1D)) {
    // ref: https://keras.io/api/datasets/mnist/
    // Deconstruct the returned Mnist struct.
   let Mnist {
    trn_img,
    trn_lbl,
    tst_img,
    tst_lbl,
    ..
} = MnistBuilder::new()
    .label_format_digit()
    .training_set_length(1_000)
    .finalize();

    let x_train = Array::from(trn_img);
    let y_train = Array::from(trn_lbl);
    let x_test = Array::from(tst_img);
    let y_test = Array::from(tst_lbl);

    ((x_train,y_train),(x_test,y_test))
}


