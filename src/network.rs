use std::{
    fmt::Debug,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    process::Output,
    usize,
};

use anyhow::Result;
use num::Float;

use crate::{
    img::Img,
    matrix::{Matrix2d, Test},
};

#[derive(Debug)]
pub struct Network<T: Clone> {
    pub input: usize,
    pub hidden: usize,
    pub output: usize,
    pub learning_rate: T,
    pub hidden_weights: Matrix2d<T>,
    pub output_weights: Matrix2d<T>,
}

impl<T: Test + Debug> Network<T> {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: T) -> Result<Self> {
        let mut hidden_weights = Matrix2d::<T>::new(hidden, input);
        hidden_weights.randomize(hidden)?;
        let mut output_weights = Matrix2d::<T>::new(output, hidden);
        output_weights.randomize(output)?;

        Ok(Self {
            input,
            hidden,
            output,
            learning_rate,
            hidden_weights,
            output_weights,
        })
    }

    pub fn train(&mut self, input: &Matrix2d<T>, output: &Matrix2d<T>) {
        let hidden_inputs = self.hidden_weights.dot_par(input);
        let hidden_outputs = hidden_inputs.apply(&sigmoid);
        let final_inputs = self.output_weights.dot_par(&hidden_outputs);
        let final_outputs = final_inputs.apply(&sigmoid);

        let output_errors = (output.clone()) - final_outputs.clone();
        let transposed_mat = self.output_weights.transpose_par();
        let hidden_errors = transposed_mat.dot_par(&output_errors);

        let sigmoid_primed_mat = sigmoid_prime(final_outputs);
        let multiplied_mat = output_errors.mul_par(&sigmoid_primed_mat);
        let transposed_mat = hidden_outputs.transpose_par();
        let dot_mat = multiplied_mat.dot_par(&transposed_mat);
        let scaled_mat = dot_mat.scale_par(self.learning_rate);
        let added_mat = scaled_mat + self.output_weights.clone();

        self.output_weights = added_mat;

        let sigmoid_primed_mat = sigmoid_prime(hidden_outputs);

        let multiplied_mat = hidden_errors.mul_par(&sigmoid_primed_mat);
        let transposed_mat = input.transpose_par();
        let dot_mat = multiplied_mat.dot_par(&transposed_mat);
        let scaled_mat = dot_mat.scale_par(self.learning_rate);
        let added_mat = scaled_mat + self.hidden_weights.clone();
        self.hidden_weights = added_mat;
    }

    pub fn train_batch_imgs(&mut self, imgs: &Vec<Img<T>>) {
        for (i, img) in imgs.iter().enumerate() {
            if i % 100 == 0 {
                println!("Img No. {i}",);
            }
            let img_data = img.matrix.flatten(crate::matrix::Axis::Row);
            let mut output = Matrix2d::new(10, 1);
            output.fill(T::zero());
            output[img.label as usize][0] = T::one();
            self.train(&img_data, &output);
        }
    }

    pub fn predict(&self, input_data: &Matrix2d<T>) -> Matrix2d<T> {
        let hidden_inputs = self.hidden_weights.dot_par(&input_data);
        let hidden_outputs = hidden_inputs.apply(&sigmoid);
        let final_inputs = self.output_weights.dot_par(&hidden_outputs);
        let final_outputs = final_inputs.apply(&sigmoid);
        let result = softmax(final_outputs);
        result
    }
    pub fn predict_img(&self, img: &Img<T>) -> Matrix2d<T> {
        let img_data = img.matrix.flatten(crate::matrix::Axis::Row);
        return self.predict(&img_data);
    }
    pub fn predict_imgs(&mut self, imgs: &Vec<Img<T>>) -> f64 {
        let mut correct = 0;
        for img in imgs {
            let prediction = self.predict_img(img);
            dbg!(&prediction);
            let guess = prediction.argmax().unwrap();
            if guess == img.label as usize {
                correct += 1;
            }
            println!("{}\nGuess: {}", img, guess);
        }

        return 1.0 / (correct as f64);
    }
}

impl<T: Float + ToString + Clone> Network<T> {
    pub fn save(&self, dirname: &str) -> Result<()> {
        fs::DirBuilder::new().recursive(true).create(dirname)?;

        let path = PathBuf::from(dirname);

        let mut descriptor = OpenOptions::new()
            .create(true)
            .write(true)
            .open(path.join("descriptor"))?;
        write!(
            descriptor,
            "{}\n{}\n{}\n",
            self.input, self.hidden, self.output
        )?;

        self.hidden_weights
            .save(path.join("hidden").to_str().unwrap())?;
        self.output_weights
            .save(path.join("output").to_str().unwrap())?;
        Ok(())
    }
    pub fn load(&mut self, dirname: &str) -> Result<()> {
        let path = PathBuf::from(dirname);

        let mut descriptor = OpenOptions::new()
            .read(true)
            .open(path.join("descriptor"))?;
        let mut content = String::new();
        descriptor.read_to_string(&mut content)?;
        let mut splitted = content.split("\n");
        self.input = splitted.next().unwrap().parse()?;
        self.hidden = splitted.next().unwrap().parse()?;
        self.output = splitted.next().unwrap().parse()?;

        self.hidden_weights = Matrix2d::load(path.join("hidden").to_str().unwrap())?;
        self.output_weights = Matrix2d::load(path.join("output").to_str().unwrap())?;

        Ok(())
    }
}

pub fn sigmoid<T: Float>(input: &T) -> T {
    T::one() / (T::one() + (-T::one() * (*input)).exp())
}
pub fn sigmoid_prime<T: Float>(m: Matrix2d<T>) -> Matrix2d<T> {
    let mut ones = Matrix2d::new(m.rows(), m.columns());
    ones.fill(T::one());
    let subtraceted = ones - m.clone();
    let multiplied = m * subtraceted;
    return multiplied;
}

pub fn softmax<T: Float>(m: Matrix2d<T>) -> Matrix2d<T> {
    let mut total = T::zero();
    for r in 0..m.rows() {
        for c in 0..m.columns() {
            total = total + m[r][c].exp();
        }
    }

    let mut mat = Matrix2d::new(m.rows(), m.columns());
    for r in 0..m.rows() {
        for c in 0..m.columns() {
            mat[r][c] = m[r][c] / total;
        }
    }
    mat
}
