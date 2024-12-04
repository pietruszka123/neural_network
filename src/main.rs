use anyhow::{anyhow, Result};
use matrix::Matrix2d;
use num::{Float, ToPrimitive};
use rand::{seq::SliceRandom, Rng};

pub mod matrix;

#[cfg(test)]
mod tests;

pub fn uniform_distribution<T: Float>(low: T, high: T) -> Result<T> {
    let difference = high - low;
    let scale: usize = 10_000;
    let scaled_difference = (T::from(scale).unwrap() * difference)
        .to_usize()
        .ok_or(anyhow!("Failed to convert into usize"))?;

    let mut rng = rand::thread_rng();

    T::from(low.to_f64().ok_or(anyhow!("Failed to convert to f64"))?
        + (1.0
            * rng.gen_range(
                0.0..scaled_difference
                    .to_f64()
                    .ok_or(anyhow!("Failed to convert to f64"))?,
            ))).ok_or(anyhow!("Failed to convert from f64"))
}

fn main() {
    let mut m = Matrix2d::<f32>::new(2, 3);
    




   

}
