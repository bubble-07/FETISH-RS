extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;
use crate::featurized_points::*;
use crate::space_info::*;
use crate::params::*;
use crate::rand_utils::*;
use crate::local_featurization_inverse_solver::*;
use crate::featurization_boundary_point_solver::*;
use crate::minimum_volume_enclosing_ellipsoid::*;
use std::rc::*;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::brent::*;

#[derive(Clone)]
pub struct Ellipsoid {
    inv_schmear : InverseSchmear
}

impl Ellipsoid {
    pub fn new(center : Array1<f32>, skew : Array2<f32>) -> Ellipsoid {
        let inv_schmear = InverseSchmear {
            mean : center,
            precision : skew
        };
        Ellipsoid {
            inv_schmear
        }
    }

    pub fn from_single_point(center : Array1<f32>) -> Ellipsoid {
        let d = center.shape()[0];
        let skew = Array::zeros((d, d));
        Ellipsoid::new(center, skew)
    }

    pub fn contains(&self, vec : &Array1<f32>) -> bool {
        self.sq_mahalanobis_dist(vec) < 1.0f32
    }

    pub fn dims(&self) -> usize {
        self.inv_schmear.mean.shape()[0]
    }

    pub fn center(&self) -> &Array1<f32> {
        &self.inv_schmear.mean
    }

    pub fn skew(&self) -> &Array2<f32> {
        &self.inv_schmear.precision
    }

    pub fn sq_mahalanobis_dist(&self, vec : &Array1<f32>) -> f32 {
        self.inv_schmear.sq_mahalanobis_dist(vec)
    }
    pub fn transform_compress(&self, mat : &Array2<f32>) -> Ellipsoid {
        let new_inv_schmear = self.inv_schmear.transform_compress(mat);
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }

    //If this is y, and we have space information space_info whose
    //featurization map is f, and a sampled collection of points in
    //the input space of the featurization map, find an ellipsoid for x
    //in f(x) = y whose image under f is approximately contained in this ellipsoid
    pub fn approx_backpropagate_through_featurization(&self, feat_points : &mut FeaturizedPoints,
                                                      x_samples : Vec<Array1<f32>>) -> Option<Ellipsoid> {
        let maybe_contained_vec = self.approx_backpropagate_through_featurization_contained_vec(feat_points, x_samples);
        match (maybe_contained_vec) {
            Option::None => Option::None,
            Option::Some(contained_vec) => Option::Some(self.approx_enclosing_ellipsoid(feat_points, &contained_vec))
        }
    }
    pub fn approx_backpropagate_through_featurization_contained_vec(&self, feat_points : &mut FeaturizedPoints,
                                                      mut x_samples : Vec<Array1<f32>>) -> Option<Array1<f32>> {
        let mut min_mahalanobis_dist = f32::INFINITY;
        let mut min_x = x_samples[0].clone();
        for x in x_samples.drain(..) {
            let y = feat_points.get_features(&x);
            let d = self.sq_mahalanobis_dist(&y);
            if (d < 1.0f32) {
                trace!("Succeeded with initial samples. Finding enclosing ellipsoid");
                return Option::Some(x);
            }
            if (d < min_mahalanobis_dist) {
                min_mahalanobis_dist = d;
                min_x = x;
            }
        }
        trace!("Initial samples were not sufficient, closest had sq dist {}", min_mahalanobis_dist);
        trace!("Running optimizer");
        //We went through all of our samples and didn't find a suitable point
        //so as a last-ditch effort, we take the best point in our samples and work
        //to minimize the Mahalanobis norm after being mapped through featurization
        //If the minimizer is sufficient for its image to be contained in this ellipsoid,
        //we'll use it, but otherwise, we'll just give up
        let space_info = feat_points.get_space_info();

        let linesearch = MoreThuenteLineSearch::new().c(MORE_THUENTE_A, MORE_THUENTE_B).unwrap();
        let solver = LBFGS::new(linesearch, LBFGS_HISTORY);

        let local_feat_inverse_solver = LocalFeaturizationInverseSolver {
            space_info : Rc::clone(&space_info),
            ellipsoid : self.clone()
        };

        let maybe_result = Executor::new(local_feat_inverse_solver, solver, min_x)
                                    .max_iters(NUM_OPT_ITERS)
                                    .run();

        match (maybe_result) {
            Result::Ok(result) => {
                if (result.state.cost < 1.0f32) {
                    //Local optimization gave us the goods, so use 'em
                    trace!("Optimization succeeded. Finding enclosing ellipsoid");
                    Option::Some(result.state.param)
                } else {
                    trace!("Optimizer failed to produce a good enough result. Sq dist: {}", result.state.cost);
                    //We tried our best, but it just wasn't good enough
                    Option::None
                }
            },
            Result::Err(err) => {
                error!("Optimizer failed: {}", err);
                Option::None
            }
        }
    }

    //Given a point whose featurization we know is within this ellipsoid,
    //get an ellipsoid about that point in the input space to the featurization map
    //such that the ellipsoid's image under featurization is contained in this ellipsoid
    pub fn approx_enclosing_ellipsoid(&self, feat_points : &mut FeaturizedPoints, x : &Array1<f32>) -> Ellipsoid {
        let mut boundary_points = Vec::new();

        let dim = x.shape()[0];
        let n_samps = dim * ENCLOSING_ELLIPSOID_DIRECTION_MULTIPLIER;

        let mut rng = rand::thread_rng();

        for i in 0..n_samps {
            let direction = gen_nsphere_random(&mut rng, dim);
            let boundary_point = self.find_boundary_point(feat_points, x, &direction);
            boundary_points.push(boundary_point);
        }
        let result_ellipsoid = minimum_volume_enclosing_ellipsoid(&boundary_points);
        result_ellipsoid
    }

    fn find_boundary_point(&self, feat_points : &mut FeaturizedPoints, 
                                  x_init : &Array1<f32>, direction : &Array1<f32>) -> Array1<f32> {
        let mut scale = ENCLOSING_ELLIPSOID_INITIAL_SCALE;

        let mut delta_x = direction.clone();
        delta_x *= ENCLOSING_ELLIPSOID_INITIAL_SCALE;

        while (scale < ENCLOSING_ELLIPSOID_MAXIMAL_SCALE) {
            scale *= ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
            delta_x *= ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
            
            let x = x_init + &delta_x;
            let y = feat_points.get_features(&x);
            if (self.sq_mahalanobis_dist(&y) >= 1.0f32) {
                //We have a bracket of the point where the mahalanobis
                //distance is exactly one, so we must do some root-finding now.
                let prev_scale = scale / ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
                let init_param = scale;
                let solver = Brent::new(prev_scale, scale, ENCLOSING_ELLIPSOID_BRENT_REL_ERROR);
                
                let problem = FeaturizationBoundaryPointSolver {
                    space_info : feat_points.get_space_info(),
                    ellipsoid : self.clone(),
                    base_point : x_init.clone(),
                    direction : direction.clone()
                };

                let opt_result = Executor::new(problem, solver, init_param)
                                      .max_iters(ENCLOSING_ELLIPSOID_BRENT_MAX_ITERS)
                                      .run();
                match (opt_result) {
                    Result::Ok(opt_result) => {
                        let opt_scale = opt_result.state.param;
                        let mut result = direction.clone();
                        result *= opt_scale;
                        result += x_init;

                        return result;
                    },
                    Result::Err(err) => {
                        error!("Boundary-finding failed: {}", err);
                        panic!();
                    }
                }
            }
        }
        error!("Infinite domains are not handled yet");
        panic!();
    }

    //If this is y, and mat is M, propagate an ellipse to x in Mx = y
    pub fn backpropagate_through_transform(&self, mat : &Array2<f32>) -> Ellipsoid {
        let u_y = &self.inv_schmear.mean;
        let s_y = &self.inv_schmear.precision;

        let mat_inv = pseudoinverse_h(mat);
        
        let u_x = mat_inv.dot(u_y);
        let s_x = mat.t().dot(s_y).dot(mat);

        let new_inv_schmear = InverseSchmear {
            mean : u_x,
            precision : s_x
        };
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }

    //If this is y, and we're given x, propagate to an ellipse on Vec(M) in Mx = y
    pub fn backpropagate_to_vectorized_transform(&self, x : &Array1<f32>) -> Ellipsoid {
        let u_y = &self.inv_schmear.mean;
        let s_y = &self.inv_schmear.precision;

        let s = x.shape()[0];
        let t = u_y.shape()[0];
        let d = s * t;

        let mut u_M_full = outer(u_y, x);
        u_M_full *= 1.0f32 / (x.dot(x));
        let u_M = u_M_full.into_shape((d,)).unwrap();

        let x_x_t = outer(x, x);
        let s_M = kron(&x_x_t, s_y);

        let new_inv_schmear = InverseSchmear {
            mean : u_M,
            precision : s_M
        };
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }
}
