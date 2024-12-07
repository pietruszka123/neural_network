use core::panic;
use std::{
    error::Error, fmt::{Debug, Display}, fs::OpenOptions, io::{Read, Write}, ops::{Add, Index, IndexMut, Mul, Sub}, str::FromStr, usize
};

use anyhow::{anyhow, Result};
use num::{Float, ToPrimitive};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::uniform_distribution;

pub fn i_to_xy(i: usize, columns: usize) -> (usize, usize) {
    return (i / columns ,i % columns, );
}

#[derive(Debug, Clone)]
pub enum Axis {
    Row,
    Collumn,
}

trait MatrixBase: Clone + Float + Sized + Send {}

#[derive(Debug, Clone)]
pub struct Matrix2d<T: Clone> {
    inner: Vec<T>,
    rows: usize,
    columns: usize,
}
impl<T: Float + Sized> Matrix2d<T> {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            inner: vec![T::nan(); rows * columns],
            rows,
            columns,
        }
    }

    pub fn fill(&mut self, n: T) {
        for i in 0..self.inner.len() {
            self.inner[i] = n;
        }
    }
    pub fn flatten(&self, axis: Axis) -> Self {
        let mut mat = match axis {
            Axis::Row => Self::new(self.columns * self.rows, 1),
            Axis::Collumn => Self::new(1, self.columns * self.rows),
        };
        for r in 0..self.rows {
            for c in 0..self.columns {
                mat.inner[r * self.columns + c] = self.inner[r * self.columns + c]
            }
        }

        mat
    }

    pub fn argmax(&self) -> Result<usize, ()> {
        if self.rows != 1 && self.columns != 1 {
            return Err(());
        }

        let mut max_value = T::min_value();

        let mut max_index = 0;

        for (i, v) in self.inner.iter().enumerate() {
            if *v > max_value {
                max_value = *v;
                max_index = i;
            }
        }

        Ok(max_index)
    }
    pub fn randomize(&mut self, n: usize) -> Result<()> {
        let min = T::from(-1.0 / n as f64).ok_or(anyhow!("Failed to convert from f64"))?;
        let max = T::from(1.0 / n as f64).ok_or(anyhow!("Failed to convert from f64"))?;
        for entry in self.inner.iter_mut() {
            *entry = uniform_distribution(min, max)?;
        }
        Ok(())
    }

    pub fn compare_dims(&self, other_matrix: &Self) -> bool {
        self.columns == other_matrix.columns && self.rows == other_matrix.rows
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn apply<F>(&self, fun: &F) -> Self
    where
        F: Fn(&T) -> T,
    {
        let mut new = Self::new(self.rows, self.columns);

        for (i, v) in self.inner.iter().enumerate() {
            new.inner[i] = fun(v);
        }

        new
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        if self.columns != rhs.rows {
            panic!("Wrong size matrices size {} != {} self columns and rhs rows lenght must be the same",self.columns,rhs.rows);
        }

        let mut new = Self::new(self.rows, rhs.columns);
        for r1 in 0..self.rows {
            for c2 in 0..rhs.columns {
                let mut sum = T::zero();
                for r2 in 0..rhs.rows {
                    sum = sum + (self[r1][r2] * rhs[r2][c2]);
                }
                new[r1][c2] = sum;
            }
        }
        new
    }

    pub fn scale(&self, n: T) -> Self {
        let mut new = Self::new(self.rows, self.columns);
        for (i, v) in self.inner.iter().enumerate() {
            new.inner[i] = (*v) * n;
        }
        new
    }

    pub fn add_scalar(&self, n: T) -> Self {
        let mut new = Self::new(self.rows, self.columns);
        for (i, v) in self.inner.iter().enumerate() {
            new.inner[i] = (*v) + n;
        }
        new
    }

    pub fn transpose(&self) -> Self {
        let mut new = Self::new(self.columns, self.rows);
        for i in 0..self.columns * self.rows {
            let (x, y) = i_to_xy(i, self.columns);

            let new_i = y * self.rows + x;

            new.inner[new_i] = self.inner[i];
        }
        new
    }
}

pub trait Test: Float + Send + Sync {}
impl<T: Float + Send + Sync> Test for T {}

#[cfg(feature = "rayon")]
impl<T: Test> Matrix2d<T> {
    pub fn apply_par<F>(&self, fun: &F) -> Self
    where
        F: (Fn(&T) -> T) + Send + Sync + 'static,
    {
        let mut new = self.clone();

        new.inner.par_iter_mut().for_each(|x| *x = fun(x));

        new
    }
    pub fn dot_par(&self, rhs: &Self) -> Self {
        if self.columns != rhs.rows {
            panic!("Wrong size matrices size {} != {} self columns and rhs rows lenght must be the same",self.columns,rhs.rows);
        }

        let mut new = Self::new(self.rows, rhs.columns);

        new.inner.par_iter_mut().enumerate().for_each(|(i, e)| {
            let (x, y) = i_to_xy(i, new.columns);
            let sum = (0..rhs.rows)
                .into_par_iter()
                .map(|r2| {
                    //println!("x: {} y: {}  r2: {} self.rows: {} self.columns: {} rhs.rows: {} rhs.columns: {} i: {}",x,y,r2,self.rows,self.columns,rhs.rows,rhs.columns,i);
                    return self[x][r2] * rhs[r2][y];
                })
                .reduce(
                    || T::zero(),
                    |x, y| {
                        return x + y;
                    },
                );
            *e = sum;
        });

        new
    }

    pub fn scale_par(&self, n: T) -> Self {
        let mut new = self.clone();
        new.inner.par_iter_mut().for_each(|x| *x = (*x) * n);
        new
    }

    pub fn add_scalar_par(&self, n: T) -> Self {
        let mut new = self.clone();
        new.inner.par_iter_mut().for_each(|x| *x = (*x) + n);
        new
    }
    pub fn transpose_par(&self) -> Self {
        let mut new = Self::new(self.columns, self.rows);

        new.inner.par_iter_mut().enumerate().for_each(|(i, v)| {
            let (x, y) = i_to_xy(i, self.rows);

            let old_i = y * self.columns + x;

            *v = self.inner[old_i];
        });

        new
    }

    pub fn mul_par(mut self,rhs: &Self) -> Self{
        panic_if_wrong_size(&self, &rhs);

        self.inner.par_iter_mut().enumerate().for_each(|(i,x)|{
            *x = (*x) * rhs.inner[i];

        });
        self
    }
    pub fn add_par(mut self,rhs: &Self) -> Self{
        panic_if_wrong_size(&self, &rhs);

        self.inner.par_iter_mut().enumerate().for_each(|(i,x)|{
            *x = (*x) + rhs.inner[i];

        });
        self
    }
}

impl<T: Float + ToString> Matrix2d<T> {
    pub fn save(&self, filename: &str) -> Result<()> {
        let mut file = OpenOptions::new().create(true).write(true).open(filename)?;
        write!(file, "{}\n{}\n", self.rows, self.columns)?;
        write!(
            file,
            "{}",
            self.inner
                .iter()
                .map(|x| { x.to_string() })
                .collect::<Vec<String>>()
                .join("\n")
        )?;
        Ok(())
    }
    pub fn load(filename: &str) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).open(filename)?;
        let mut b = String::new();
        file.read_to_string(&mut b)?;

        let mut splitted = b.split("\n");
        let rows: usize = splitted.next().unwrap().parse()?;
        let columns: usize = splitted.next().unwrap().parse()?;

        let mut inner = Vec::new();

        for line in splitted {
            inner.push(T::from(line.trim().parse::<f64>()?).unwrap());
        }

        Ok(Self {
            inner,
            rows,
            columns,
        })
    }
}

impl<T: Float + ToString> Display for Matrix2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        dbg!(self.rows, self.columns);
        let mut out = String::new();

        for r in 0..self.rows {
            if r != 0 {
                out += " ";
            }
            for c in 0..self.columns {
                let i = (r * self.columns) + c;
                out += &self.inner[i].to_string();
                if c != self.columns - 1 {
                    out += ",";
                }
            }
            if r != self.rows - 1 {
                out += "\n";
            }
        }
        write!(f, "[{}]", out)
    }
}

impl<T: Float> Index<usize> for Matrix2d<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        return &self.inner[index * self.columns..(index + 1) * self.columns];
    }
}
impl<T: Float> IndexMut<usize> for Matrix2d<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.inner[index * self.columns..(index + 1) * self.columns];
    }
}

fn panic_if_wrong_size<T: Float>(m1: &Matrix2d<T>, m2: &Matrix2d<T>) {
    if !m1.compare_dims(&m2) {
        panic!(
            "Matrices dimensions are not the same {} != {}, {} != {}",
            m1.columns, m2.columns, m1.rows, m2.rows
        );
    }
}

//TODO: check if correct
impl<T: Float> Mul for Matrix2d<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        panic_if_wrong_size(&self, &rhs);
        //I hope compiler will optimize this
        let mut new = Self::new(self.rows, rhs.columns);

        for (i, v) in self.inner.iter().enumerate() {
            let v2 = rhs.inner[i];
            new.inner[i] = (*v) * v2;
        }
        new
    }
}

impl<T: Float> Add for Matrix2d<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        panic_if_wrong_size(&self, &rhs);
        let mut new = Self::new(self.rows, self.columns);

        for (i, v) in self.inner.iter().enumerate() {
            let v2 = rhs.inner[i];
            new.inner[i] = (*v) + v2;
        }
        new
    }
}

impl<T: Float> Sub for Matrix2d<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        panic_if_wrong_size(&self, &rhs);
        let mut new = Self::new(self.rows, self.columns);

        for (i, v) in self.inner.iter().enumerate() {
            let v2 = rhs.inner[i];
            new.inner[i] = (*v) - v2;
        }
        new
    }
}
