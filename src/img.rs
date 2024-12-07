use std::{fmt::Display, fs::File, io::Read};

use anyhow::Result;
use num::Float;

use crate::matrix::Matrix2d;

#[derive(Debug)]
pub struct Img<T: Clone> {
    pub matrix: Matrix2d<T>,
    pub label: u32,
}
impl<T:Clone + Float> Img<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            matrix: Matrix2d::new(rows, columns),
            label: 0,
        }
    }
}

pub fn csv_to_imgs<T: Clone + Float>(file: &mut File, number_of_images: usize) -> Result<Vec<Img<T>>> {
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let lines = content.split('\n');

    let mut imgs = Vec::new();
    for line in lines.skip(1).take(number_of_images) {
        let mut nums = line.split(',');
        let mut img = Img::new(28, 28);
        img.label = nums.next().unwrap().parse::<u32>()?;
        for (i, num) in nums.enumerate() {
            let rows = img.matrix.rows();
            let n = match num.trim().parse::<i32>() {
                Ok(n) => Some(n),
                Err(e) => match e.kind() {
                    std::num::IntErrorKind::Empty => None,
                    _ => return Err(e.into()),
                },
            };
            if let Some(n) = n {
                img.matrix[i / rows][i % rows] = (T::from(n).unwrap()) / T::from(256.0).unwrap();
            }
        }
        imgs.push(img);
    }
    Ok(imgs)
}

const SHADES: [char; 5] = [' ', '░', '▒','▓', '█'];

fn normalize<T: Float>(v: T) -> T {
    return v * T::from(4.).unwrap();
}

impl<T:Clone + Float> Display for Img<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        for r in 0..self.matrix.rows() {
            for c in 0..self.matrix.columns() {
                let v = self.matrix[r][c];

                out += &SHADES[normalize(v).ceil().to_usize().unwrap()].to_string();
            }

            if r != self.matrix.rows() - 1 {
                out += "\n"
            }
        }

        write!(f, "label: {}\n{}", self.label, out)
    }
}
