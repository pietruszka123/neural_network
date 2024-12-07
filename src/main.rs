use std::fs::OpenOptions;

use anyhow::{anyhow, Result};
use img::{csv_to_imgs, Img};
use matrix::Matrix2d;
use network::Network;
use num::{Float, ToPrimitive};
use rand::{seq::SliceRandom, Rng};

pub mod img;
pub mod matrix;

pub mod network;
#[cfg(test)]
mod tests;

pub fn uniform_distribution<T: Float>(low: T, high: T) -> Result<T> {
    let difference = high - low;
    let scale: usize = 10_000;
    let scaled_difference = (T::from(scale).unwrap() * difference)
        .to_usize()
        .ok_or(anyhow!("Failed to convert into usize"))?;

    let mut rng = rand::thread_rng();

    T::from(
        low.to_f64().ok_or(anyhow!("Failed to convert to f64"))?
            + (1.0
                * (rng.gen_range(
                    0.0..scaled_difference
                        .to_f64()
                        .ok_or(anyhow!("Failed to convert to f64"))?,
                ) / scale as f64)),
    )
    .ok_or(anyhow!("Failed to convert from f64"))
}

fn main() {
    let mut m = Matrix2d::<f32>::new(2, 3);

    let mut file = OpenOptions::new()
        .read(true)
        .open("data/mnist_train.csv")
        .unwrap();


    //let number_imgs = 10000;
    //let imgs = csv_to_imgs::<f64>(&mut file, number_imgs).unwrap();
    //let mut net = Network::new(784, 300, 10, 0.1).unwrap();
    //
    //net.train_batch_imgs(&imgs);
    //net.save("./testing_net").unwrap();

    let number_imgs = 100;
    let imgs = csv_to_imgs(&mut file, number_imgs).unwrap();
    let mut net = Network::new(1, 1, 1, 0.).unwrap();
    net.load("./testing_net").unwrap();

    net.predict_imgs(&imgs);
}
