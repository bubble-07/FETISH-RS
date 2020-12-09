extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::test_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::array_utils::*;
use crate::ellipsoid::*;
use crate::ellipsoid_sampler::*;
use crate::pseudoinverse::*;
use crate::params::*;

//Ported from https://gist.github.com/raluca-san/5d8bd4dcb3278b8e2f6dfa959614f9c8
pub fn minimum_volume_enclosing_ellipsoid(points : &Vec<Array1<f32>>) -> Ellipsoid {
    trace!("Finding minimum volume enclosing ellipsoid of {} points", points.len());
    let N = points.len();
    let d = points[0].shape()[0];

    let mut point_array = Array::zeros((N, d));
    let mut Q = Array::ones((N, d + 1));
    for i in 0..N {
        for j in 0..d {
            Q[[i, j]] = points[i][j];
            point_array[[i, j]] = points[i][j];
        }
    }
    Q = Q.t().to_owned();

    let mut error = 1.0f32;

    let mut u = Array::ones((N,));
    u *= (1.0f32 / (N as f32));

    while (error > ENCLOSING_ELLIPSOID_TOLERANCE) {
        let Q_t_row_scaled = scale_rows(&Q.t().to_owned(), &u);
        let X = Q.dot(&Q_t_row_scaled);
        let X_inv = pseudoinverse_h(&X);
        let M = Q.t().dot(&X_inv).dot(&Q);
        let M_diag = M.diag();
        let (max_M_diag_index, max_M_diag_value) = max_index_and_value(&M_diag);
        let step_size = (max_M_diag_value - (d as f32) - 1.0f32) / 
                        (((d as f32) + 1.0f32) * (max_M_diag_value - 1.0f32));
        let mut new_u = u.clone();
        new_u *= (1.0f32 - step_size);
        new_u[[max_M_diag_index,]] += step_size;
        error = sq_vec_dist(&new_u, &u).sqrt();
        u = new_u;
    }
    let center = u.dot(&point_array);

    let points_row_scaled = scale_rows(&point_array, &u);
    let points_u_inner = point_array.t().dot(&points_row_scaled);
    let center_outer = outer(&center, &center);

    let skew_inv = &points_u_inner - &center_outer;
    let mut skew = pseudoinverse_h(&skew_inv);
    skew *= (1.0f32 / (d as f32));

    trace!("Minimum volume enclosing ellipsoid found");

    Ellipsoid::new(center, skew)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ellipsoid_reconstruction() {
        let num_samps = 50;
        let dim = 2;
        let mut rng = rand::thread_rng();
        let mut points = Vec::new();
        let ellipsoid = random_ellipsoid(dim);
        let ellipsoid_sampler = EllipsoidSampler::new(&ellipsoid);
        for _ in 0..num_samps {
            let boundary_vec = ellipsoid_sampler.sample_boundary(&mut rng);
            points.push(boundary_vec);
            let contained_vec = ellipsoid_sampler.sample(&mut rng);
            points.push(contained_vec);
        }
        let reconstructed = minimum_volume_enclosing_ellipsoid(&points);
        assert_equal_matrices_to_within(ellipsoid.skew(), reconstructed.skew(), 0.2f32);
        assert_equal_vectors_to_within(ellipsoid.center(), reconstructed.center(), 0.1f32);
    }

    #[test]
    fn test_unit_ellipsoid_from_coordinate_axes() {
        let dim = 3;
        let mut points = Vec::new();
        for i in 0..dim {
            let mut pos = Array::zeros((dim,));
            pos[[i,]] = 1.0f32;
            points.push(pos);
            let mut neg = Array::zeros((dim,));
            neg[[i,]] = -1.0f32;
            points.push(neg);
        }
        let eye = Array::eye(dim);
        let zero = Array::zeros((dim,));

        let ellipsoid = minimum_volume_enclosing_ellipsoid(&points);
        assert_equal_matrices(ellipsoid.skew(), &eye);
        assert_equal_vectors(ellipsoid.center(), &zero);
    }
}
