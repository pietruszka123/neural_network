use std::{cmp::min, fmt::Display, fs::File, io::Read, num::ParseIntError};

use anyhow::Result;

use crate::matrix::{self, Matrix2d};

#[derive(Debug)]
pub struct Img {
    pub matrix: Matrix2d<f32>,
    pub label: u32,
}
impl Img {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            matrix: Matrix2d::new(rows, columns),
            label: 0,
        }
    }
}

pub fn csv_to_imgs(file: &mut File, number_of_images: usize) -> Result<Vec<Img>> {
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
                img.matrix[i / rows][i % rows] = (n as f32) / 256.0;
            }
        }
        imgs.push(img);
    }
    Ok(imgs)
}

const SHADES: [char; 5] = [' ', '░', '▒','▓', '█'];

fn normalize(v: f32) -> f32 {
    return v * 4.;
}

impl Display for Img {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        for r in 0..self.matrix.rows() {
            for c in 0..self.matrix.columns() {
                let v = self.matrix[r][c];

                out += &SHADES[normalize(v).ceil() as usize].to_string();
            }

            if r != self.matrix.rows() - 1 {
                out += "\n"
            }
        }

        write!(f, "{}\n{}", self.label, out)
    }
}
