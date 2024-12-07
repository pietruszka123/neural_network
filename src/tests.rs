use crate::matrix::Matrix2d;

#[test]
fn dot() {
    let mut m = Matrix2d::<f64>::new(2, 3);
    let d = [3., 2., 4., 9., 7., 6.];
    for r in 0..m.rows() {
        for c in 0..m.columns() {
            m[r][c] = d[r * m.columns() + c];
        }
    }

    let mut m2 = Matrix2d::<f64>::new(3, 2);
    let d = [1., 5., 3., 9., 7., 4.];
    for r in 0..m2.rows() {
        for c in 0..m2.columns() {
            m2[r][c] = d[r * m2.columns() + c];
        }
    }
    println!("{m}\n{m2}");

    let r = m.dot(&m2);
    let rp = m.dot_par(&m2);
    println!("{r}");
    println!("{rp}");

    let t = m2.transpose();
    let t2 = m2.transpose_par();
    println!("{t}");
    println!("{t2}");

}
