extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;
use crate::space_info::*;
use crate::params::*;
use crate::rand_utils::*;
use crate::local_featurization_inverse_solver::*;
use crate::featurization_boundary_point_solver::*;
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

    pub fn contains(&self, vec : &Array1<f32>) -> bool {
        self.mahalanobis_dist(vec) < 1.0f32
    }

    pub fn center(&self) -> &Array1<f32> {
        &self.inv_schmear.mean
    }

    pub fn skew(&self) -> &Array2<f32> {
        &self.inv_schmear.precision
    }

    pub fn mahalanobis_dist(&self, vec : &Array1<f32>) -> f32 {
        self.inv_schmear.mahalanobis_dist(vec)
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
    pub fn approx_backpropagate_through_featurization(&self, space_info : Rc<SpaceInfo>, 
                                                      mut x_samples : Vec<Array1<f32>>) -> Option<Ellipsoid> {
        let mut min_mahalanobis_dist = f32::INFINITY;
        let mut min_x = x_samples[0].clone();
        for x in x_samples.drain(..) {
            let y = space_info.get_features(&x);
            let d = self.mahalanobis_dist(&y);
            if (d < 1.0f32) {
                return self.approx_enclosing_ellipsoid(space_info, &x);
            }
            if (d < min_mahalanobis_dist) {
                min_mahalanobis_dist = d;
                min_x = x;
            }
        }
        //We went through all of our samples and didn't find a suitable point
        //so as a last-ditch effort, we take the best point in our samples and work
        //to minimize the Mahalanobis norm after being mapped through featurization
        //If the minimizer is sufficient for its image to be contained in this ellipsoid,
        //we'll use it, but otherwise, we'll just give up

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
                    self.approx_enclosing_ellipsoid(space_info, &result.state.param)
                } else {
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
    fn approx_enclosing_ellipsoid(&self, space_info : Rc<SpaceInfo>, x : &Array1<f32>) -> Option<Ellipsoid> {
        let dim = x.shape()[0];
        let n_samps = dim * ENCLOSING_ELLIPSOID_DIRECTION_MULTIPLIER;

        let mut rng = rand::thread_rng();

        for i in 0..n_samps {
            let direction = gen_nsphere_random(&mut rng, dim);
            let boundary_point = self.find_boundary_point(space_info.clone(), x, &direction);
        }
        //TODO: Call out to a minimum enclosed ellipsoid routine
        Option::None
    }

    fn find_boundary_point(&self, space_info : Rc<SpaceInfo>, 
                                  x_init : &Array1<f32>, direction : &Array1<f32>) -> Array1<f32> {
        let mut scale = ENCLOSING_ELLIPSOID_INITIAL_SCALE;

        let mut delta_x = direction.clone();
        delta_x *= ENCLOSING_ELLIPSOID_INITIAL_SCALE;

        while (scale < ENCLOSING_ELLIPSOID_MAXIMAL_SCALE) {
            scale *= ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
            delta_x *= ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
            
            let x = x_init + &delta_x;
            let y = space_info.get_features(&x);
            if (self.mahalanobis_dist(&y) >= 1.0f32) {
                //We have a bracket of the point where the mahalanobis
                //distance is exactly one, so we must do some root-finding now.
                let prev_scale = scale / ENCLOSING_ELLIPSOID_GROWTH_FACTOR;
                let init_param = scale;
                let solver = Brent::new(prev_scale, scale, ENCLOSING_ELLIPSOID_BRENT_REL_ERROR);
                
                let problem = FeaturizationBoundaryPointSolver {
                    space_info : space_info,
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
